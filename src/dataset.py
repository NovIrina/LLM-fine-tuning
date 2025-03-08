from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_dataset(path: Path, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Loads and prepares the dataset for processing.

    Args:
        path (Path): The path to the dataset.
        tokenizer (AutoTokenizer): The tokenizer to be used on the dataset.

    Returns:
        Dict[str, Any]: The processed dataset.
    """
    try:
        dataset = load_dataset(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {path}: {e}")

    column_names = list(dataset["train"].features)
    dataset = dataset.map(
        lambda batch: process_batch(batch, tokenizer, column_names),
        batched=True,
        batch_size=None,
        num_proc=1,
        remove_columns=column_names,
    )
    return dataset


def process_input_ids(input_ids: List[int], tokenizer: AutoTokenizer) -> List[int]:
    """
    Processes labels for the input IDs.

    Args:
        input_ids (List[int]): The input IDs from the tokenizer.
        tokenizer (AutoTokenizer): The tokenizer used to retrieve the pad token ID.

    Returns:
        List[int]: The labels created from the input IDs.
    """
    return input_ids[1:] + [tokenizer.pad_token_id]


def process_batch(
    batch: Dict[str, List[str]], tokenizer: AutoTokenizer, column_names: List[str]
) -> Dict[str, Any]:
    """
    Processes a single batch of data.

    Args:
        batch (Dict[str, List[str]]): The batch of data to process.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding.
        column_names (List[str]): The names of the columns in the dataset.

    Returns:
        Dict[str, Any]: The processed batch with input IDs and labels.
    """
    encodings = tokenizer(
        [
            f"{batch[column_names[0]][i]} {batch[column_names[1]][i]} {tokenizer.eos_token}"
            for i in range(len(batch[column_names[0]]))
        ],
        padding="max_length",
        max_length=512,
        truncation=True,
    )

    batch["labels"] = [
        process_input_ids(input_ids, tokenizer) for input_ids in encodings["input_ids"]
    ]
    return batch
