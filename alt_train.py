# coding=utf-8
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
import logging
import os
import sys
import gc
import torch

from transformers import (
    HfArgumentParser,
    set_seed,
)
from Src.Args import Arguments, expend_args
from Src.CHMM.NHMMTrainingPreparation import prepare_chmm_training

logger = logging.getLogger(__name__)


def main():
    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]
    expend_args(args=args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # --- setup logging ---
    logging.basicConfig(
        format="[%(levelname)s - %(name)s] - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info("Training/evaluation parameters %s", args)

    # --- initialize result-storing file ---
    denoising_model = args.denoising_model if args.denoising_model else 'true'
    nhmm_output = os.path.join(
        args.output_dir, f"{args.dataset_name}-{denoising_model}-{args.seed}_results"
    )

    # --- Set seed ---
    set_seed(args.seed)

    # --- setup Neural HMM training functions ---
    nhmm_trainer = prepare_chmm_training(args=args)

    # --- train Neural HMM ---
    logger.info(" --- starting Neural HMM training process --- ")

    micro_results = nhmm_trainer.train()

    results = nhmm_trainer.test()
    logger.info("[INFO] test results:")
    for k, v in results.items():
        if 'entity' in k:
            logger.info("  %s = %s", k, v)

    logger.info(" --- Neural HMM training is successfully finished --- ")
    logger.info(f" --- Writing results to {nhmm_output} ---")

    with open(nhmm_output + '-1.txt', 'w') as f:
        for i, micro_result in enumerate(micro_results):
            f.write(f"[Epoch] {i + 1}\n")
            for k, v in micro_result.items():
                if 'entity' in k:
                    f.write("%s = %s\n" % (k, v))
        f.write(f"[Test]\n")
        for k, v in results.items():
            if 'entity' in k:
                f.write("%s = %s\n" % (k, v))

    logger.info(f" --- Results written --- ")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
