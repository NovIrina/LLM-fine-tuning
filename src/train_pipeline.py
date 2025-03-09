"""
This module contains the training pipeline for the model.
"""
from functools import partial
from typing import Any, Dict, Tuple

import evaluate
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments, default_data_collator

from src.arguments import ProjectArguments
from src.dataset import prepare_dataset
from src.model import load_model
from src.tokenizer import load_tokenizer


def compute_metrics(metric: Any, eval_predictions: Tuple) -> Dict[str, float]:
    """
    Computes metrics for the model's predictions.

    Args:
        metric (Any): The metric to compute.
        eval_predictions (Tuple): Tuple containing predictions and labels.

    Returns:
        Dict[str, float]: Computed metrics.
    """
    predictions, labels = eval_predictions
    labels = labels[:, 1:].reshape(-1)
    predictions = predictions[:, :-1].reshape(-1)
    return metric.compute(predictions=predictions, references=labels)


def train_model(arguments: ProjectArguments) -> None:
    """
    Trains the model using the specified configurations and datasets.
    """
    model = load_model(arguments.path_to_model)
    tokenizer = load_tokenizer(arguments.path_to_tokenizer)
    dataset = prepare_dataset(arguments.path_to_dataset, tokenizer)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=arguments.lora_rank,
        lora_alpha=arguments.lora_alpha,
        lora_dropout=arguments.lora_dropout,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    metric = evaluate.load("accuracy")
    compute_metrics_func = partial(compute_metrics, metric)

    training_args = TrainingArguments(
        output_dir="outputs/checkpoints/",
        evaluation_strategy="epoch",
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=4000,
        weight_decay=0.01,
        warmup_steps=500,
        optim="adamw_torch",
        label_smoothing_factor=0.1,
        learning_rate=6e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        eval_steps=1,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        use_mps_device=True,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_func,
    )

    print("Starting training")
    trainer.train()
    print("Finished training")
    trainer.save_model("outputs/checkpoints")
