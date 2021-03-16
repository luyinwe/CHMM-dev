# coding=utf-8
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
import logging
import os
import sys

from transformers import (
    HfArgumentParser,
    set_seed,
)
from Src.Args import Arguments, expend_args
from Src.Assist.SrcPerformance import source_performance

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

    # --- Set seed ---
    set_seed(args.seed)

    # --- Get Source Performances ---
    source_performance(args)


if __name__ == "__main__":
    main()
