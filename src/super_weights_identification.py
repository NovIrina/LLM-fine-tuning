# pylint: disable=redefined-outer-name, protected-access
"""
This script visualizes the super weights of the model by plotting the maximum activations.
"""
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tap import Tap
from transformers import AutoTokenizer

from src.model import load_model
from src.tokenizer import load_tokenizer


class SuperWeightsIdentification(Tap):
    """
    Defines the command-line arguments for the script.
    """
    path_to_model: Path
    path_to_tokenizer: Path


def get_max_activations(
    model: torch.nn.Module, inputs: Dict[str, torch.Tensor]
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """
    Retrieves the maximum activations from the model's layers for given inputs.

    Args:
        model (torch.nn.Module): The model to analyze.
        inputs (Dict[str, torch.Tensor]): The inputs to the model.

    Returns:
        Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
            Two dictionaries containing the largest input and output activations for each layer.
    """
    all_activations = {}

    def get_activations(layer_index: int, module_name: str):
        """
        Creates a hook to capture activations from a specific layer/module.
        """

        def hook(
            _: torch.nn.Module,
            inputs: Tuple[torch.Tensor],
            outputs: Tuple[torch.Tensor],
        ):
            all_activations[f"Layer {layer_index}, module: {module_name} output"] = (
                outputs[0]
            )
            all_activations[f"Layer {layer_index}, module: {module_name} input"] = (
                inputs[0]
            )

        return hook

    for index, layer in enumerate(model._modules["transformer"].h):
        layer.attn.c_attn.register_forward_hook(get_activations(index, "attn.c_attn"))
        layer.attn.c_proj.register_forward_hook(get_activations(index, "attn.c_proj"))
        layer.mlp.c_proj.register_forward_hook(get_activations(index, "mlp.c_proj"))
        layer.mlp.c_fc.register_forward_hook(get_activations(index, "mlp.c_fc"))

    model.eval()
    with torch.no_grad():
        model(**inputs)

    pattern = r"Layer (?P<layer>\d+), module: (?P<module>[a-zA-Z\._]+)"
    largest_values_inputs = {i: {} for i in range(len(model._modules["transformer"].h))}
    largest_values_outputs = {
        i: {} for i in range(len(model._modules["transformer"].h))
    }

    for key, value in all_activations.items():
        tensor = value.detach().cpu()
        tensor_abs = tensor.view(-1).abs().float()
        max_value, _ = torch.max(tensor_abs, 0)

        if "input" in key:
            match = re.match(pattern, key)
            if match:
                largest_values_inputs[int(match.group("layer"))][
                    match.group("module")
                ] = max_value.item()

        if "output" in key:
            match = re.match(pattern, key)
            if match:
                largest_values_outputs[int(match.group("layer"))][
                    match.group("module")
                ] = max_value.item()

    return largest_values_inputs, largest_values_outputs


def make_plot(dataframe: pd.DataFrame, label: str) -> None:
    """
    Generates and saves a plot of activation values over layer numbers.

    Args:
        dataframe (pd.DataFrame): The data containing layer numbers and activation values.
        label (str): The label for the plot, indicating whether it's for inputs or outputs.
    """
    plt.figure(figsize=(10, 6))
    for module in dataframe.columns[1:]:
        plt.plot(dataframe["layer_number"], dataframe[module], marker="o", label=module)

    plt.title("Super Weights inside Different Layers")
    plt.xlabel("Layer Number")
    plt.ylabel(f"Activation Values {label}")
    plt.legend()
    plt.grid()
    plt.savefig(f"outputs/figures/super_weights_{label}.pdf")


def visualize_super_weights(model: torch.nn.Module, tokenizer: AutoTokenizer) -> None:
    """
    Visualizes the super weights of the model by plotting the maximum activations.

    Args:
        model (torch.nn.Module): The model to analyze.
        tokenizer (Any): The tokenizer used to preprocess input text.
    """
    text = "Cats are incredibly intelligent animals."
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    largest_values_inputs, largest_values_outputs = get_max_activations(model, inputs)

    df_inputs = pd.DataFrame.from_dict(
        largest_values_inputs, orient="index"
    ).reset_index()
    df_inputs.rename(columns={"index": "layer_number"}, inplace=True)
    make_plot(df_inputs, "inputs")

    df_outputs = pd.DataFrame.from_dict(
        largest_values_outputs, orient="index"
    ).reset_index()
    df_outputs.rename(columns={"index": "layer_number"}, inplace=True)
    make_plot(df_outputs, "outputs")


if __name__ == "__main__":
    parser = SuperWeightsIdentification().parse_args()
    model = load_model(parser.path_to_model)
    tokenizer = load_tokenizer(parser.path_to_tokenizer)
    visualize_super_weights(model, tokenizer)
