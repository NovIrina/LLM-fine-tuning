"""
This module is used to train and evaluate the model.
"""
import argparse
from pathlib import Path
from typing import List, Optional, Union

from train_eval_pipeline.arguments import TrainEvalArguments
from train_eval_pipeline.constants import PATH_TO_DATASET, PATH_TO_MODEL, PATH_TO_TOKENIZER
from train_eval_pipeline.eval_pipeline import eval_model
from train_eval_pipeline.modes import UsageModes
from train_eval_pipeline.train_pipeline import train_model


def parse_arguments():
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_model",
        type=Path,
        default=PATH_TO_MODEL,
        required=True,
        help="Path to model",
    )
    parser.add_argument(
        "--path_to_tokenizer",
        type=Path,
        default=PATH_TO_TOKENIZER,
        required=True,
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--path_to_dataset",
        type=Path,
        default=PATH_TO_DATASET,
        required=True,
        help="Path to dataset",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=4, required=True, help="Rank of LoRA adapters"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        required=True,
        help="Alpha of LoRA adapters",
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, required=True, help="LoRA dropout"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=Union[str, List[str]],
        default="all-linear",
        required=True,
        help="Target modules to apply LoRA adapters",
    )
    parser.add_argument(
        "--lora_layers_to_transform",
        type=Optional[List[str]],
        default=None,
        required=True,
        help="Layers to apply LoRA adapters",
    )
    parser.add_argument(
        "--mode",
        type=UsageModes,
        default=UsageModes.TRAINING_AND_EVALUATION,
        required=True,
        help="Mode to run pipeline"
    )

    return parser.parse_args()


if __name__ == "main":
    args = parse_arguments()
    config = TrainEvalArguments(
        path_to_model=args.path_to_model,
        path_to_peft_model=args.path_to_peft_model,
        path_to_tokenizer=args.path_to_tokenizer,
        path_to_dataset=args.path_to_dataset,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_layers_to_transform=args.lora_layers_to_transform,
    )
    if args.mode is UsageModes.TRAINING:
        train_model(config)
    elif args.mode is UsageModes.EVALUATION:
        eval_model(config)
    else:
        train_model(config)
        eval_model(config)
