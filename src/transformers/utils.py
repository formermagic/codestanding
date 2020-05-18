from typing import Any, Optional

import numpy as np
import torch


def safe_round(number: Any, ndigits: int) -> float:
    if hasattr(number, "__round__"):
        return round(number, ndigits)
    elif torch.is_tensor(number) and number.numel() == 1:
        return safe_round(number.item(), ndigits)
    elif np.ndim(number) == 0 and hasattr(number, "item"):
        return safe_round(number.item(), ndigits)
    else:
        return number


def get_perplexity(
    loss: Optional[Any], ndigits: int = 2, base: int = 2
) -> torch.FloatTensor:
    if loss is None:
        # type: ignore
        return torch.FloatTensor([0.0])
    try:
        # type: ignore
        return torch.FloatTensor([safe_round(base ** loss, ndigits)])
    except OverflowError:
        # type: ignore
        return torch.FloatTensor([float("inf")])
