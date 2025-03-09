"""
This module contains the evaluation pipeline for the model.
"""
from typing import Tuple

import evaluate
import pandas as pd
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from train_eval_pipeline.arguments import TrainEvalArguments
from train_eval_pipeline.model import load_model, load_peft_model
from train_eval_pipeline.tokenizer import load_tokenizer
from train_eval_pipeline.utils import get_torch_device


def compute_metrics(y_pred: list, y_true: list) -> None:
    """
    Computes and prints BLEU and perplexity scores.

    Args:
        y_pred (list): List of predicted sentences.
        y_true (list): List of reference sentences.
    """
    metric = evaluate.load("bleu")
    metric.add_batch(predictions=y_pred, references=y_true)
    bleu = float(metric.compute()["bleu"]) * 100
    print(f"BLEU score: {bleu}")

    perplexity_metric = load_metric("perplexity")
    perplexity = perplexity_metric.compute(predictions=y_pred, references=y_true)
    print(f"Perplexity score: {perplexity}")


def batch_generator(
    dataset: Dataset, batch_size: int, tokenizer: AutoTokenizer
):
    """
    Generates batches of tokenized queries from the dataset.

    Args:
        dataset (Dataset): The dataset to generate batches from.
        batch_size (int): The size of each batch.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenization.

    Yields:
        Tuple: A tokenized batch and the original batch.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    for batch in loader:
        queries = list(batch["meaning_representation"])
        tokenized_query = tokenizer(queries, return_tensors="pt", padding=True).to(
            get_torch_device()
        )
        yield tokenized_query, batch


def get_batch_generator(
    dataset: Dataset, batch_size: int, tokenizer: AutoTokenizer
) -> Tuple[iter, int]:
    """
    Creates a batch generator for the validation dataset.

    Args:
        dataset (Dataset): The dataset to use for validation.
        batch_size (int): The size of each batch.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenization.

    Returns:
        Tuple[iter, int]: A generator for batches and the total number of unique queries.
    """
    dataset = Dataset.from_dict(
        {
            "meaning_representation": dataset["validation"].unique(
                "meaning_representation"
            )
        }
    )
    return (
        batch_generator(dataset, batch_size, tokenizer),
        len(dataset["meaning_representation"]),
    )


def validation_loop(  # pylint: disable=too-many-locals
    batches: iter,
    model: AutoModel,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    num_eval_steps: int,
) -> Tuple[list, list, list]:
    """
    Performs validation on the model using the provided batches.

    Args:
        batches (iter): The generated batches for evaluation.
        model (Any): The model to evaluate.
        dataset (Dataset): The dataset containing reference sentences.
        tokenizer (Any): The tokenizer to decode predictions.
        num_eval_steps (int): The number of evaluation steps to perform.

    Returns:
        Tuple[list, list, list]: Lists of queries, predicted sentences, and reference sentences.
    """
    generation_settings = {
        "max_new_tokens": 64,
        "num_beams": 1,  # Disable beam search by setting to 1
        "do_sample": False,  # Use greedy decoding
        "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.0,
    }

    all_queries = []
    all_references = []
    all_predicted = []

    model = model.to(get_torch_device())

    for input_batch, batch in tqdm(batches, total=num_eval_steps):
        input_batch = input_batch.to(get_torch_device())
        output = model.generate(**input_batch, **generation_settings)

        predictions = []
        for i in range(output.size(0)):
            generated_tokens = output[i, input_batch["input_ids"][i].size(0) :]
            predictions.append(generated_tokens)

        batch_pred = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        for predicted_sentence, meaning_representation in zip(
            batch_pred, batch["meaning_representation"]
        ):
            filtered_dataset = dataset.filter(
                lambda x: x["meaning_representation"] == meaning_representation  # pylint: disable=cell-var-from-loop
            )
            for reference_sentence in filtered_dataset["validation"]["human_reference"]:
                all_queries.append(meaning_representation)
                all_predicted.append(predicted_sentence)
                all_references.append(reference_sentence)

    pd.DataFrame(
        {
            "queries": all_queries,
            "predictions": all_predicted,
            "references": all_references,
        }
    ).to_csv("outputs/predictions/pred_vs_ref.csv", index=False)

    return all_queries, all_predicted, all_references


def eval_model(arguments: TrainEvalArguments) -> None:
    """
    Evaluates the model and computes metrics.
    """
    model = load_model(arguments.path_to_model)
    peft_model = load_peft_model(model, arguments.path_to_peft_model)
    tokenizer = load_tokenizer(arguments.path_to_tokenizer)
    dataset = load_dataset(arguments.path_to_dataset)
    batches = get_batch_generator(
        dataset, arguments.eval_batch_size, tokenizer, arguments.device
    )

    num_iterations = batches[1] // 8
    all_queries, all_predicted, all_references = validation_loop(
        batches[0], peft_model, dataset, tokenizer, num_iterations
    )
    compute_metrics(all_predicted, all_references)
