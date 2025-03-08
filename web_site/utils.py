import torch


def get_torch_device() -> torch.device:
    """
    Get the device to use for the model.

    Returns:
        torch.device: The device to use.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
