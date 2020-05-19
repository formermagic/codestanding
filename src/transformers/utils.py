from typing import Any, Optional, cast

import numpy as np
import torch


def safe_round(number: Any, ndigits: int) -> float:
    round_number: float
    if hasattr(number, "__round__"):
        round_number = round(number, ndigits)
    elif torch.is_tensor(number) and number.numel() == 1:
        round_number = safe_round(number.item(), ndigits)
    elif np.ndim(number) == 0 and hasattr(number, "item"):
        round_number = safe_round(number.item(), ndigits)
    else:
        round_number = number
    return round_number


# pylint: disable=not-callable
def get_perplexity(
    loss: Optional[Any], ndigits: int = 2, base: int = 2
) -> torch.FloatTensor:
    ppl_tensor: torch.Tensor
    if loss is None:
        ppl_tensor = torch.tensor([0.0])
    try:
        ppl_tensor = torch.tensor([safe_round(base ** loss, ndigits)])
    except OverflowError:
        ppl_tensor = torch.tensor([float("inf")])

    return cast(torch.FloatTensor, ppl_tensor.float())
