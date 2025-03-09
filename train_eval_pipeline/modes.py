from enum import Enum


class UsageModes(Enum):
    TRAINING = "train"
    EVALUATION = "eval"
    TRAINING_AND_EVALUATION = "train_eval"
