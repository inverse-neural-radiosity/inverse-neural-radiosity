from pathlib import Path

import mitsuba as mi
import torch
import torch.nn.functional as F
from torch import Tensor


def save_image(root: str, name: str, formats: list[str], image):
    """Save image to multiple formats, each in a sub folder"""
    assert all(fmt in {"png", "exr"} for fmt in formats)

    root: Path = Path(root)
    for fmt in formats:
        folder = root / fmt
        folder.mkdir(parents=True, exist_ok=True)
        write_bitmap(str(folder / f"{name}.{fmt}"), image)


def convert_to_bitmap(data, gamma_correction, uint8_srgb=True):
    """
    Convert the RGB image in `data` to a `Bitmap`. `uint8_srgb` defines whether
    the resulting bitmap should be translated to a uint8 sRGB bitmap.
    """

    if isinstance(data, mi.Bitmap):
        bitmap = data
    else:
        if isinstance(data, Tensor):
            data = data.detach().cpu().numpy()
        bitmap = mi.Bitmap(data)

    if uint8_srgb:
        bitmap = bitmap.convert(
            mi.Bitmap.PixelFormat.RGBA,
            mi.Struct.Type.UInt8,
            gamma_correction,
        )

    return bitmap


def write_bitmap(filename, data, tonemap=True,  write_async=True, quality=-1):
    """
    Write the RGB image in `data` to a PNG/EXR/.. file.
    """
    uint8_srgb = Path(filename).suffix in {".png", ".jpg", ".jpeg", ".webp"}

    bitmap = convert_to_bitmap(data, tonemap, uint8_srgb)

    if write_async:
        bitmap.write_async(filename, quality=quality)
    else:
        bitmap.write(filename, quality=quality)


def find_nonzero_batches(im, c):
    """
    Find offsets (i,j) on an image that given a crop size c, lead to nonzero batches or with more than half the pixels on
    Zero pixel is defined as alpha channel = 0
    im: [H,W,4]
    c: int
    """
    assert len(im.shape) == 3
    assert im.shape[-1] == 4

    h, w = im.shape[0], im.shape[1]
    mask = (im[:, :, -1] > 0).float()

    # run a 2d covultion using an all one kernel
    in_put = mask.unsqueeze(0).unsqueeze(0).to(im.device)
    weight = torch.ones(1, 1, c, c).to(im.device)
    convolved_2 = F.conv2d(in_put, weight, padding='same').squeeze(dim=1)

    # shift the results since the convolution stores the results in the center of the kernel,
    # but we need it be on top left

    d = (c-1)//2
    result = convolved_2[:, d:h-(c-d), d:w-(c-d)]
    result = result.squeeze(dim=0)

    # check if it has at least 500 pixels on
    nonzero_inds = (result > (c*c*0.5)).nonzero()
    return nonzero_inds
