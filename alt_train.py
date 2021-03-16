# coding=utf-8
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
import logging
import os
import sys
import gc
import torch

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from Src.Args import Arguments, expend_args
from Src.NHMM.NHMMTrainingPreparation import prepare_nhmm_training

logger = logging.getLogger(__name__)


def main():
    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, nhmm_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, nhmm_args, training_args = parser.parse_args_into_dataclasses()
    expend_args(args=training_args, args=nhmm_args, args=data_args)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # --- setup logging ---
    logging.basicConfig(
        format="[%(levelname)s - %(name)s] - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # --- initialize result-storing file ---
    denoising_model = data_args.denoising_model if data_args.denoising_model else 'true'
    bert_output = os.path.join(
        training_args.output_dir, f"{data_args.dataset_name}-{denoising_model}-{training_args.seed}-bert_results"
    )
    nhmm_output = os.path.join(
        training_args.output_dir, f"{data_args.dataset_name}-{denoising_model}-{training_args.seed}_results"
    )
    if os.path.exists(bert_output):
        os.remove(bert_output)

    # --- Set seed ---
    set_seed(training_args.seed)

    # --- setup Neural HMM training functions ---
    nhmm_trainer = prepare_nhmm_training(
        nhmm_args=nhmm_args, data_args=data_args, training_args=training_args
    )

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
