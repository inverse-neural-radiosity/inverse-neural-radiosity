import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def check_gradients(params: list[torch.Tensor]):
    grad_mags = []
    for p in params:
        g = p._grad  # pylint: disable=protected-access
        if g is None:
            continue
        grad_mags.append(g.norm().cpu().numpy())
    grad_mags = np.array(grad_mags)

    if len(grad_mags) == 0:
        raise ValueError("No gradient presents in any of the tensors!")
    logger.info(f"grad norm mean {grad_mags.mean()}, min {grad_mags.min()}, max {grad_mags.max()}")
