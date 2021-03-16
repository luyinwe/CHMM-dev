import torch
import torch.nn as nn
from Core.Util import log_matmul, log_maxmul, validate_prob, logsumexp
from Core.Constants import OntoNotes_BIO
from Core.Data import label_to_span
from typing import Optional


class NeuralModule(nn.Module):
    def __init__(self,
                 d_emb,
                 n_hidden,
                 n_src,
                 n_obs):
        super(NeuralModule, self).__init__()

        self.n_hidden = n_hidden
        self.n_src = n_src
        self.n_obs = n_obs
        self.neural_transition = nn.Linear(d_emb, self.n_hidden * self.n_hidden)
        self.neural_emissions = nn.ModuleList([
            nn.Linear(d_emb, self.n_hidden * self.n_obs) for _ in range(self.n_src)
        ])
        # self.neural_emission = nn.Linear(d_emb, self.n_hidden * self.n_hidden * self.n_src)

        self._init_parameters()

    def forward(self,
                embs: torch.Tensor,
                temperature: Optional[int] = 1.0):
        batch_size, max_seq_length, _ = embs.size()
        trans_temp = self.neural_transition(embs).view(
            batch_size, max_seq_length, self.n_hidden, self.n_hidden
        )
        nn_trans = torch.softmax(trans_temp / temperature, dim=-1)

        nn_emiss = torch.stack([torch.softmax(emiss(embs).view(
                batch_size, max_seq_length, self.n_hidden, self.n_obs
            ) / temperature, dim=-1) for emiss in self.neural_emissions]).permute(1, 2, 0, 3, 4)
        return nn_trans, nn_emiss

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.neural_transition.weight.data, gain=nn.init.calculate_gain('relu'))
        for emiss in self.neural_emissions:
            nn.init.xavier_uniform_(emiss.weight.data, gain=nn.init.calculate_gain('relu'))


class NeuralHMM(nn.Module):
    """
    Neural Hidden Markov Model.
    (For now, discrete obs_set only.)
    - forward(): computes the log probability of an observation sequence.
    - viterbi(): computes the most likely state sequence.
    """

    def __init__(self,
                 args,
                 state_prior=None,
                 trans_matrix=None,
                 emiss_matrix=None):
        super(NeuralHMM, self).__init__()

        self.d_emb = args.d_emb  # embedding dimension
        self.n_src = args.n_src
        self.n_obs = args.n_obs  # number of possible obs_set
        self.n_hidden = args.n_hidden  # number of states

        self.trans_weight = args.trans_nn_weight
        self.emiss_weight = args.emiss_nn_weight

        self.device = args.device

        self.nn_module = NeuralModule(d_emb=self.d_emb, n_hidden=self.n_hidden, n_src=self.n_src, n_obs=self.n_obs)

        # initialize unnormalized state-prior, transition and emission matrices
        self._initialize_model(
            state_prior=state_prior, trans_matrix=trans_matrix, emiss_matrix=emiss_matrix
        )

    def _initialize_model(self,
                          state_prior: torch.Tensor,
                          trans_matrix: torch.Tensor,
                          emiss_matrix: torch.Tensor
                          ):

        if state_prior is None:
            priors = torch.zeros(self.n_hidden, device=self.device) + 1E-3
            priors[0] = 1
            self.state_priors = nn.Parameter(torch.log(priors))
        else:
            # TODO: beware of the -inf case
            state_prior.to(self.device)
            priors = validate_prob(state_prior, dim=0)
            self.state_priors = nn.Parameter(torch.log(priors))

        if trans_matrix is None:
            self.unnormalized_trans = nn.Parameter(torch.randn(self.n_hidden, self.n_hidden, device=self.device))
        else:
            trans_matrix.to(self.device)
            trans_matrix = validate_prob(trans_matrix)
            # We may want to use softmax later, so we put here a log to counteract the effact
            self.unnormalized_trans = torch.log(trans_matrix).to(self.device)

        if emiss_matrix is None:
            self.unnormalized_emiss = nn.Parameter(
                torch.zeros(self.n_src, self.n_hidden, self.n_obs, device=self.device)
            )
        else:
            emiss_matrix.to(self.device)
            emiss_matrix = validate_prob(emiss_matrix)
            # We may want to use softmax later, so we put here a log to counteract the effact
            self.unnormalized_emiss = nn.Parameter(torch.log(emiss_matrix))

        print("[INFO] model initialized!")

        return None

    def _initialize_states(self,
                           embs: torch.Tensor,
                           obs: torch.Tensor,
                           temperature: Optional[int] = 1.0):
        # normalize and put the probabilities into the log domain
        batch_size, max_seq_length, n_src, _ = obs.size()
        self.log_state_priors = torch.log_softmax(self.state_priors / temperature, dim=-1)
        trans = torch.softmax(self.unnormalized_trans / temperature, dim=-1)
        emiss = torch.softmax(self.unnormalized_emiss / temperature, dim=-1)

        # get neural transition and emission matrices
        # TODO: we can add layer-norm later to see what happens
        _, nn_emiss = self.nn_module(embs)

        # TODO: the coefficients are subject to change
        self.log_trans = torch.log(trans.view(
            1, 1, self.n_hidden, self.n_hidden
        ).repeat(batch_size, max_seq_length, 1, 1))
        self.log_emiss = torch.log((1-self.emiss_weight) * emiss + self.emiss_weight * nn_emiss)

        # Calculate the emission probabilities in one time, so that we don't have to compute this repeatedly
        # log-domain subtract is regular-domain divide
        self.log_emiss_probs = log_matmul(
            self.log_emiss, torch.log(obs).unsqueeze(-1)
        ).squeeze().sum(dim=-2)

        self.log_alpha = torch.zeros([batch_size, max_seq_length, self.n_hidden], device=self.device)
        self.log_beta = torch.zeros([batch_size, max_seq_length, self.n_hidden], device=self.device)
        # Gamma can be readily computed and need no initialization
        self.log_gamma = None
        # only values in 1:max_seq_length are valid. The first state is a dummy
        self.log_xi = torch.zeros([batch_size, max_seq_length, self.n_hidden, self.n_hidden], device=self.device)
        return None

    def _forward_step(self, t):
        # initial alpha state
        if t == 0:
            log_alpha_t = self.log_state_priors + self.log_emiss_probs[:, t, :]
        # do the forward step
        else:
            log_alpha_t = self.log_emiss_probs[:, t, :] + \
                          log_matmul(self.log_alpha[:, t - 1, :].unsqueeze(1), self.log_trans[:, t, :, :]).squeeze()

        # normalize the result
        normalized_log_alpha_t = log_alpha_t - log_alpha_t.logsumexp(dim=-1, keepdim=True)
        return normalized_log_alpha_t

    def _backward_step(self, t):
        # do the backward step
        # beta is not a distribution, so we do not need to normalize it
        log_beta_t = log_matmul(
            self.log_trans[:, t, :, :],
            (self.log_emiss_probs[:, t, :] + self.log_beta[:, t + 1, :]).unsqueeze(-1)
        ).squeeze()
        return log_beta_t

    def _forward_backward(self, seq_lengths):
        max_seq_length = seq_lengths.max().item()
        # calculate log alpha
        for t in range(0, max_seq_length):
            self.log_alpha[:, t, :] = self._forward_step(t)

        # calculate log beta
        # The last beta state beta[:, -1, :] = log1 = 0, so no need to re-assign the value
        for t in range(max_seq_length - 2, -1, -1):
            self.log_beta[:, t, :] = self._backward_step(t)
        # shift the output (since beta is calculated in backward direction,
        # we need to shift each instance in the batch according to its length)
        shift_distances = seq_lengths - max_seq_length
        self.log_beta = torch.stack(
            [torch.roll(beta, s.item(), 0) for beta, s in zip(self.log_beta, shift_distances)]
        )
        return None

    def _compute_xi(self, t):
        temp_1 = self.log_emiss_probs[:, t, :] + self.log_beta[:, t, :]
        temp_2 = log_matmul(self.log_alpha[:, t-1, :].unsqueeze(-1), temp_1.unsqueeze(1))
        log_xi_t = self.log_trans[:, t, :, :] + temp_2
        return log_xi_t

    def _expected_complete_log_likelihood(self, seq_lengths):
        batch_size = len(seq_lengths)
        max_seq_length = seq_lengths.max().item()

        # calculate expected sufficient statistics: gamma_t(j) = P(z_t = j|x_{1:T})
        self.log_gamma = self.log_alpha + self.log_beta
        # normalize as gamma is a distribution
        log_gamma = self.log_gamma - self.log_gamma.logsumexp(dim=-1, keepdim=True)

        # calculate expected sufficient statistics: psi_t(i, j) = P(z_{t-1}=i, z_t=j|x_{1:T})
        for t in range(1, max_seq_length):
            self.log_xi[:, t, :, :] = self._compute_xi(t)
        stabled_norm_term = logsumexp(self.log_xi[:, 1:, :, :].view(batch_size, max_seq_length-1, -1), dim=-1)\
            .view(batch_size, max_seq_length-1, 1, 1)
        log_xi = self.log_xi[:, 1:, :, :] - stabled_norm_term

        # calculate the expected complete data log likelihood
        log_prior = torch.sum(torch.exp(log_gamma[:, 0, :]) * self.log_state_priors, dim=-1)
        log_prior = log_prior.mean()
        # sum over j, k
        # TODO: it seems that this is a bug... I should have used logtran[:, 1:, :, :] instead of the current one?
        log_tran = torch.sum(torch.exp(log_xi) * self.log_trans[:, 1:, :, :], dim=[-2, -1])
        # sum over valid time steps, and then average over batch. Note this starts from t=2
        log_tran = torch.mean(torch.stack([inst[:length].sum() for inst, length in zip(log_tran, seq_lengths-1)]))
        # same as above
        log_emis = torch.sum(torch.exp(log_gamma) * self.log_emiss_probs, dim=-1)
        log_emis = torch.mean(torch.stack([inst[:length].sum() for inst, length in zip(log_emis, seq_lengths)]))
        log_likelihood = log_prior + log_tran + log_emis

        return log_likelihood

    def forward(self, emb, obs, seq_lengths):
        """
        obs : IntTensor of shape (batch size, max_length, n_src)
        lengths : IntTensor of shape (batch size)

        Compute log p(obs) for each example in the batch.
        lengths = max_seq_length of each example
        """
        # the row of obs should be one-hot or at least sum to 1
        # assert (obs.sum(dim=-1) == 1).all()

        batch_size, max_seq_length, n_src, n_obs = obs.size()
        assert n_obs == self.n_obs
        assert n_src == self.n_src

        # Initialize alpha, beta and xi
        self._initialize_states(embs=emb, obs=obs)
        self._forward_backward(seq_lengths=seq_lengths)
        log_likelihood = self._expected_complete_log_likelihood(seq_lengths=seq_lengths)
        return log_likelihood, (self.log_trans, self.log_emiss)

    def viterbi(self, emb, obs, seq_lengths):
        """
        obs : IntTensor of shape (batch size, max_len, n_src)
        seq_lengths : IntTensor of shape (batch size)

        Find argmax_z log p(z|obs) for each (obs) in the batch.
        """
        batch_size = len(seq_lengths)
        max_seq_length = seq_lengths.max().item()

        # initialize states
        self._initialize_states(embs=emb, obs=obs)
        # maximum probabilities
        log_delta = torch.zeros([batch_size, max_seq_length, self.n_hidden], device=self.device)
        # most likely previous state on the most probable path to z_t = j. a[0] is undefined.
        pre_states = torch.zeros([batch_size, max_seq_length, self.n_hidden], dtype=torch.long, device=self.device)

        # the initial delta state
        log_delta[:, 0, :] = self.log_state_priors + self.log_emiss_probs[:, 0, :]
        for t in range(1, max_seq_length):
            # udpate delta and a. It does not matter where we put the emission probabilities
            max_log_prob, argmax_val = log_maxmul(
                log_delta[:, t-1, :].unsqueeze(1),
                self.log_trans[:, t, :, :] + self.log_emiss_probs[:, t, :].unsqueeze(1)
            )
            log_delta[:, t, :] = max_log_prob.squeeze()
            pre_states[:, t, :] = argmax_val.squeeze()

        # The terminal state
        batch_max_log_prob = list()
        batch_z_t_star = list()

        for l_delta, length in zip(log_delta, seq_lengths):
            max_log_prob, z_t_star = l_delta[length-1, :].max(dim=-1)
            batch_max_log_prob.append(max_log_prob)
            batch_z_t_star.append(z_t_star)

        # Trace back
        batch_z_star = [[z_t_star.item()] for z_t_star in batch_z_t_star]
        for p_states, z_star, length in zip(pre_states, batch_z_star, seq_lengths):
            for t in range(length-2, -1, -1):
                z_t = p_states[t+1, z_star[0]].item()
                z_star.insert(0, z_t)

        return batch_z_star, batch_max_log_prob

    def annotate(self,
                 emb,
                 obs,
                 seq_lengths,
                 label_set=OntoNotes_BIO):
        batch_label_indices, batch_probs = self.viterbi(emb, obs, seq_lengths)
        batch_labels = [[label_set[lb_index] for lb_index in label_indices]
                        for label_indices in batch_label_indices]
        batch_spans = list()
        for labels, probs in zip(batch_labels, batch_probs):
            batch_spans.append(label_to_span(labels))

        return batch_spans
