import os
import torch
from Src.DataAssist import extract_sequence, converse_ontonote_to_conll
from Src.CHMM.CHMMData import Dataset, collate_fn
from Src.CHMM.CHMMTrain import CHMMTrainer


def prepare_chmm_training(args) -> CHMMTrainer:

    # ----- construct dataset -----
    ontonote_anno_scheme = True if (args.dataset_name == 'Co03' and not args.converse_first) or \
                                   args.ontonote_anno_scheme else False

    args.output_dir = args.output_dir
    
    exp_train_sents, train_embs, train_lbs, train_weak_lbs = load_features(
        'train', args, ontonote_anno_scheme
    )
    exp_dev_sents, dev_embs, dev_lbs, dev_weak_lbs = load_features(
        'dev', args, ontonote_anno_scheme
    )
    exp_test_sents, test_embs, test_lbs, test_weak_lbs = load_features(
        'test', args, ontonote_anno_scheme
    )

    train_dataset = Dataset(
        text=exp_train_sents,
        embs=train_embs,
        obs=train_weak_lbs,
        lbs=train_lbs
    )
    dev_dataset = Dataset(
        text=exp_dev_sents,
        embs=dev_embs,
        obs=dev_weak_lbs,
        lbs=dev_lbs
    )
    test_dataset = Dataset(
        text=exp_test_sents,
        embs=test_embs,
        obs=test_weak_lbs,
        lbs=test_lbs
    )

    args.d_emb = train_embs[0].size(-1)
    _, args.n_src, args.n_obs = train_weak_lbs[0].size()
    args.n_hidden = args.n_obs

    if not args.use_src_weights:
        src_weights = None
    else:
        src_weights = list()
        for k in args.src_to_keep:
            src_weights.append(args.src_weights[k])

    # ----- initialize training process -----
    trainer = CHMMTrainer(
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        test_dataset=test_dataset,
        collate_fn=collate_fn,
        src_weights=src_weights
    )

    return trainer


def load_features(partition, data_args, ontonote_anno_scheme):
    if partition == 'train':
        data_name = data_args.train_name
        emb_name = data_args.train_emb
    elif partition == 'dev':
        data_name = data_args.dev_name
        emb_name = data_args.dev_emb
    elif partition == 'test':
        data_name = data_args.test_name
        emb_name = data_args.test_emb
    else:
        raise ValueError
    
    data = torch.load(os.path.join(data_args.data_dir, data_name))
    sents = data['sentences']
    embs = torch.load(os.path.join(data_args.data_dir, emb_name))
    annotations = data['annotations']
    lbs = data['labels']
    exp_sents = [["[CLS]"] + sent for sent in sents]
    if data_args.converse_first:
        annotations = converse_ontonote_to_conll(data_args, annotations)
    weak_lbs = [extract_sequence(
        s, a, sources=data_args.src_to_keep, label_indices=data_args.lbs2idx, ontonote_anno_scheme=ontonote_anno_scheme
    ) for s, a in zip(sents, annotations)]

    return exp_sents, embs, lbs, weak_lbs
