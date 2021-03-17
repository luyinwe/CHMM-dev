import os
import torch
import pprint
import pandas as pd

from Src.Utils import get_results


def source_performance(args):
    pp = pprint.PrettyPrinter(indent=4)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    # ----- load data -----
    test_data = torch.load(os.path.join(args.data_dir, args.test_name))

    # ----- construct dataset -----
    sents = test_data['sentences']
    annotations = test_data['annotations']

    true_spans = test_data['labels']

    weak_spans = dict()
    for source in args.src_to_keep:
        src_anno_list = list()
        for annotation in annotations:
            anno_dict = dict()
            for k, v in annotation[source].items():
                if isinstance(v, str):
                    pass
                elif isinstance(v, tuple) or isinstance(v, list):
                    v = v[0][0]
                if args.dataset_name == 'Co03':
                    pred_lb = v
                    norm_lb = args.mappings.get(pred_lb, pred_lb)
                    if norm_lb in args.lbs:
                        anno_dict[k] = norm_lb
                else:
                    anno_dict[k] = v
            src_anno_list.append(anno_dict)
        weak_spans[source] = src_anno_list

    results = dict()
    for k, spans in weak_spans.items():
        src_results = get_results(spans, true_spans, sents, all_labels=args.lbs)
        for key, value in src_results['micro'].items():
            if 'entity' not in key:
                continue
            if key not in results:
                results[key] = list()
            results[key].append(value)
        print(k)
        pp.pprint(src_results)

    df = pd.DataFrame(data=results, index=weak_spans.keys())
    df.index.name = 'function name'
    df.to_csv(os.path.join(args.output_dir, f"log.{args.dataset_name}.source-performance.csv"))
