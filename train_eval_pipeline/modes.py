"""
This module contains usage modes of main.py.
"""
from enum import Enum


class UsageModes(Enum):
    """
    UsageModes of main.py.
    """
    TRAINING = "train"
    EVALUATION = "eval"
    TRAINING_AND_EVALUATION = "train_eval"
