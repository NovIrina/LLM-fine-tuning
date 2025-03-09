"""
This module contains the definition of the ProjectArguments dataclass.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from train_eval_pipeline.constants import (
    PATH_TO_DATASET,
    PATH_TO_MODEL,
    PATH_TO_PEFT_MODEL,
    PATH_TO_TOKENIZER,
)


@dataclass
class TrainEvalArguments:
    """
    The arguments for the project.
    """

    path_to_model: Path = (PATH_TO_MODEL,)
    path_to_peft_model: Path = (PATH_TO_PEFT_MODEL,)
    path_to_tokenizer: Path = (PATH_TO_TOKENIZER,)
    path_to_dataset: Path = (PATH_TO_DATASET,)
    lora_rank: int = (4,)
    lora_alpha: int = (32,)
    lora_dropout: float = (0.1,)
    lora_target_modules: Union[str, List[str]] = ("all-linear",)
    lora_layers_to_transform: Optional[List[str]] = (None,)
    validation_batch_size: int = (8,)
