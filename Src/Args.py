import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional
from Src.Constants import *


@dataclass
class Arguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    output_dir: str = field(
        metadata={"help": "The output dir."}
    )
    dataset_name: str = field(
        metadata={"help": "The name of the dataset."}
    )
    train_name: Optional[str] = field(
        default='', metadata={'help': 'training data name'}
    )
    dev_name: Optional[str] = field(
        default='', metadata={'help': 'development data name'}
    )
    test_name: Optional[str] = field(
        default='', metadata={'help': 'test data name'}
    )
    denoising_model: Optional[str] = field(
        default=None,
        metadata={"help": "From which weak model the scores are generated."}
    )
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    converse_first: bool = field(
        default=False, metadata={'help': 'converse the annotation space before (True) or after training'}
    )
    model_reinit: bool = field(
        default=False, metadata={'help': 're-initialized BERT model before each training stage/loop'}
    )
    update_embeddings: bool = field(
        default=False, metadata={'help': 'whether update embeddings during mixed training process'}
    )
    redistribute_confidence: bool = field(
        default=False, metadata={'help': 'whether to make the CHMM output sharper'}
    )
    phase2_train_epochs: int = field(
        default=20, metadata={'help': 'phase 2 fine-tuning epochs'}
    )
    true_lb_ratio: float = field(
        default=0, metadata={'help': 'What is the ratio of true label to use'}
    )
    trans_nn_weight: float = field(
        default=1.0, metadata={'help': 'the weight of neural part in the transition matrix'}
    )
    emiss_nn_weight: float = field(
        default=1.0, metadata={'help': 'the weight of neural part in the emission matrix'}
    )
    denoising_epoch: int = field(
        default=15, metadata={'help': 'number of denoising model training epochs'}
    )
    denoising_pretrain_epoch: int = field(
        default=5, metadata={'help': 'number of denoising model pre-train training epochs'}
    )
    chmm_tolerance_epoch: int = field(
        default=10, metadata={"help": "How many tolerance epochs before quiting training"}
    )
    retraining_loops: int = field(
        default=10, metadata={"help": "How many self-training (denoising-training) loops to adopt"}
    )
    hmm_lr: float = field(
        default=0.01, metadata={'help': 'learning rate of the hidden markov part'}
    )
    nn_lr: float = field(
        default=0.001, metadata={'help': 'learning rate of the neural part of the Neural HMM'}
    )
    denoising_batch_size: int = field(
        default=128, metadata={'help': 'denoising model training batch size'}
    )
    obs_normalization: bool = field(
        default=False, metadata={'help': 'whether normalize observations'}
    )
    ontonote_anno_scheme: bool = field(
        default=False, metadata={'help': 'whether to use ontonote annotation scheme'}
    )
    use_src_weights: bool = field(
        default=False, metadata={'help': 'whether to use source weights'}
    )
    use_src_attention_weights: bool = field(
        default=False, metadata={'help': 'whether to calculate attention weights for each source'}
    )
    device: str = field(
        default='cuda', metadata={'help': 'the device you want to use'}
    )
    seed: int = field(
        default=0, metadata={'help': 'random seed'}
    )


def expend_args(args):
    if not args.data_dir:
        args.data_dir = os.path.join(ROOT_DIR, '../data', args.dataset_name)
    else:
        args.data_dir = os.path.join(ROOT_DIR, args.data_dir, args.dataset_name)
    if not args.train_name:
        args.train_name = args.dataset_name + '-linked-train.pt'
    if not args.dev_name:
        args.dev_name = args.dataset_name + '-linked-dev.pt'
    if not args.test_name:
        args.test_name = args.dataset_name + '-linked-test.pt'

    args.train_emb = args.train_name.replace('linked', 'emb')
    args.dev_emb = args.dev_name.replace('linked', 'emb')
    args.test_emb = args.test_name.replace('linked', 'emb')

    if args.dataset_name == 'Laptop':
        args.lbs = LAPTOP_LABELS
        args.bio_lbs = LAPTOP_BIO
        args.lbs2idx = LAPTOP_INDICES
        args.src = LAPTOP_SOURCE_NAMES
        args.src_to_keep = LAPTOP_SOURCES_TO_KEEP
        args.src_priors = LAPTOP_SOURCE_PRIORS
        args.src_weights = LAPTOP_SOURCE_WEIGHTS
    elif args.dataset_name == 'NCBI':
        args.lbs = NCBI_LABELS
        args.bio_lbs = NCBI_BIO
        args.lbs2idx = NCBI_INDICES
        args.src = NCBI_SOURCE_NAMES
        args.src_to_keep = NCBI_SOURCES_TO_KEEP
        args.src_priors = NCBI_SOURCE_PRIORS
    elif args.dataset_name == 'BC5CDR':
        args.lbs = BC5CDR_LABELS
        args.bio_lbs = BC5CDR_BIO
        args.lbs2idx = BC5CDR_INDICES
        args.src = BC5CDR_SOURCE_NAMES
        args.src_to_keep = BC5CDR_SOURCES_TO_KEEP
        args.src_priors = BC5CDR_SOURCE_PRIORS
    elif args.dataset_name == 'Co03':
        args.lbs = CoNLL_LABELS
        args.bio_lbs = OntoNotes_BIO if not args.converse_first else CoNLL_BIO
        args.lbs2idx = OntoNotes_INDICES if not args.converse_first else CoNLL_INDICES
        args.src = CoNLL_SOURCE_NAMES
        args.src_to_keep = CoNLL_SOURCE_TO_KEEP
        args.src_priors = CoNLL_SOURCE_PRIORS if not args.converse_first else CONLL_SRC_PRIORS
        args.mappings = CoNLL_MAPPINGS
        args.prior_name = 'CoNLL03-init-stat-all.pt'
    else:
        json_name = os.path.join(args.data_dir, f'{args.dataset_name}-metadata.json')
        with open(json_name, 'r') as f:
            data = json.load(f)
        args.lbs = data['labels']
        if 'source-labels' not in data.keys():
            args.bio_lbs = ["O"] + ["%s-%s" % (bi, label) for label in args.lbs for bi in "BI"]
        else:
            args.bio_lbs = ["O"] + ["%s-%s" % (bi, label) for label in data['source-labels'] for bi in "BI"]
        args.lbs2idx = {label: i for i, label in enumerate(args.bio_lbs)}
        args.src = data['sources']
        args.src_to_keep = data['sources-to-keep'] if 'sources-to-keep' in data.keys() else data['sources']
        args.src_priors = data['priors'] if 'priors' in data.keys() else \
            {src_name: {lb: (0.8, 0.8) for lb in args.lbs} for src_name in args.src_to_keep}
        args.mappings = data['mapping'] if 'mapping' in data.keys() else None

    args.device = torch.device('cuda') if torch.cuda.is_available() and args.device == 'cuda' else torch.device('cpu')
