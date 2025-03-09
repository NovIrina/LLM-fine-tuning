"""
This module contains functions for loading tokenizers.
"""
from pathlib import Path

from transformers import AutoTokenizer

from train_eval_pipeline.constants import PATH_TO_TOKENIZER


def load_tokenizer(path: Path = PATH_TO_TOKENIZER) -> AutoTokenizer:
    """
    Loads a tokenizer from a specified path.

    Args:
        path (str): The model identifier from Hugging Face's model hub.

    Returns:
        AutoTokenizer: The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token_id = 0
    return tokenizer
