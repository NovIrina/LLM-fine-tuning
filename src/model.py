"""
This module contains the function to load a pre-trained model.
"""
from pathlib import Path

from transformers import AutoModelForCausalLM

from src.constants import PATH_TO_MODEL


def load_model(
    path_to_save: Path, path_to_load: Path = PATH_TO_MODEL
) -> AutoModelForCausalLM:
    """
    Loads a pre-trained model.

    If the model exists at the specified path to save, it loads it from there;
    otherwise, it loads from the default path and saves it to the specified path.

    Args:
        path_to_save (Path): The path where the model should be saved or loaded from.
        path_to_load (str): The path to load the model from if not found at path_to_save.

    Returns:
        AutoModelForCausalLM: The loaded causal language model.
    """
    if path_to_save.exists():
        try:
            pretrained_model = AutoModelForCausalLM.from_pretrained(str(path_to_save))
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path_to_save}: {e}") from e
    else:
        try:
            pretrained_model = AutoModelForCausalLM.from_pretrained(path_to_load)
            pretrained_model.save_pretrained(path_to_save)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path_to_load}: {e}") from e

    return pretrained_model
