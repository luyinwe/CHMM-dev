import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self,
                 temperature: float,
                 attn_dropout: Optional[float] = 0.1):
        """
        :param temperature: sqrt(d_k)
        :param attn_dropout: Drop out ratio
        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 d_k: Optional[int] = 64,
                 d_v: Optional[int] = 64,
                 dropout: Optional[float] = 0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_normal_(self.w_qs.weight)
        nn.init.xavier_normal_(self.w_ks.weight)
        nn.init.xavier_normal_(self.w_vs.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.layer_norm = nn.LayerNorm(d_model)

        # TODO: made a change here to have smaller output
        self.fc = nn.Linear(n_head * d_v, n_head * d_v)
        self.qfc = nn.Linear(d_model, n_head * d_v)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.layer_norm(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        q, attn = self.attention(q, k, v, mask=mask)

        q = q.view(n_head, sz_b, len_q, d_v)
        q = q.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        q = self.dropout(self.fc(q))
        # q = self.layer_norm(q + residual)
        q += self.dropout(self.qfc(residual))

        return q, attn


class NeuralModule(nn.Module):
    def __init__(self,
                 d_emb,
                 n_hidden,
                 n_src,
                 n_obs,
                 use_src_attention_weights: Optional[bool] = False,
                 n_head: Optional[int] = 2,
                 d_kv: Optional[int] = 64):
        super(NeuralModule, self).__init__()

        self.n_hidden = n_hidden
        self.n_src = n_src
        self.n_obs = n_obs
        self.neural_transition = nn.Linear(d_emb, self.n_hidden * self.n_hidden)
        self.neural_emissions = nn.ModuleList([
            nn.Linear(d_emb, self.n_hidden * self.n_obs) for _ in range(self.n_src)
        ])

        if use_src_attention_weights:
            self.attention_layer = MultiHeadAttention(
                d_model=d_emb,
                n_head=n_head,
                d_k=d_kv,
                d_v=d_kv
            )
            self.attn_fc_k = nn.Linear(n_head * d_kv, self.n_src)
            self.softmax = nn.Softmax(dim=-1)

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

    def attention_forward(self,
                          embs: torch.Tensor
                          ):
        assert self.attention_layer, "Attention layer is not defined!"

        attn_mask = embs.sum(dim=-1) != 0
        pad_mask = ~attn_mask.unsqueeze(1).expand(-1, attn_mask.size(-1), -1)
        attn_embs, _ = self.attention_layer(
            q=embs,
            k=embs,
            v=embs,
            mask=pad_mask
        )
        attn_embs *= attn_mask.unsqueeze(-1)

        src_weiths = self.attn_fc_k(attn_embs)
        src_weiths *= attn_mask.unsqueeze(-1)

        return self.softmax(src_weiths)

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.neural_transition.weight.data, gain=nn.init.calculate_gain('relu'))
        for emiss in self.neural_emissions:
            nn.init.xavier_uniform_(emiss.weight.data, gain=nn.init.calculate_gain('relu'))
