import mitsuba as mi
import torch
import torch.nn.functional as F
from torch import Tensor


def compute_psnr(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute PSNR of torch images, values in range [0, 1]
    """
    mse = F.mse_loss(a, b)
    return 10 * torch.log10(1 / mse)


def compute_mape(pred: Tensor, gt: Tensor) -> Tensor:
    return torch.mean(torch.abs(pred - gt) / (gt + 0.01))


@torch.no_grad()
def compute_metrics_torch(pred: Tensor, gt: Tensor) -> dict[str, float]:
    pred = pred[..., :3]
    gt = gt[..., :3]
    return {
        "mse": float(F.mse_loss(pred, gt)),
        "mape": float(compute_mape(pred, gt)),
        "psnr": float(compute_psnr(pred, gt)),
    }


def compute_metrics(pred, gt) -> dict[str, float]:
    pred = _convert_to_tensor(pred)
    gt = _convert_to_tensor(gt)
    return compute_metrics_torch(pred, gt)


def _convert_to_tensor(x):
    if isinstance(x, Tensor):
        return x
    if isinstance(x, mi.Bitmap):
        return mi.TensorXf(x).torch()
    if isinstance(x, mi.TensorXf):
        return x.torch()
    raise ValueError(f"Unknown type {type(x)}")
