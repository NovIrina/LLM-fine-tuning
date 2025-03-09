"""
This module contains the function to load a pre-trained model.
"""

from pathlib import Path

from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM

from train_eval_pipeline.constants import PATH_TO_MODEL, PATH_TO_SAVE_MODEL
from train_eval_pipeline.utils import get_torch_device


def load_model(
    path_to_save: Path = PATH_TO_SAVE_MODEL, path_to_load: Path = PATH_TO_MODEL
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
        pretrained_model = AutoModelForCausalLM.from_pretrained(str(path_to_save))
    else:
        pretrained_model = AutoModelForCausalLM.from_pretrained(path_to_load)
        pretrained_model.save_pretrained(path_to_save)

    return pretrained_model


def load_peft_model(model: AutoModel, path: Path) -> PeftModel:
    """
    Loads a PEFT model from the specified path.

    Args:
        model (AutoModel): The base model to enhance with PEFT.
        path (Path): The path to load the PEFT model from.

    Returns:
        PeftModel: The loaded PEFT model.
    """
    model = PeftModel.from_pretrained(model, path)
    model = model.to(get_torch_device())
    model.eval()
    return model
