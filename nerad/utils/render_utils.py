from pathlib import Path
from typing import Union

import drjit as dr
import mitsuba as mi
import torch

from nerad.model.config import RenderingConfig
from nerad.utils.image_utils import save_image


def mis_weight(pdf_a, pdf_b):
    pdf_a *= pdf_a
    pdf_b *= pdf_b
    w = pdf_a / (pdf_a + pdf_b)
    return dr.detach(dr.select(dr.isfinite(w), w, 0))


def process_nerad_output(img):
    residual = img[:, :, -3:]
    LHS = img[:, :, -7:-3]
    RHS = img[:, :, :-7]
    return residual, LHS, RHS


def render_and_save_image(
    folder: Path,
    name: str,
    scene: mi.Scene,
    integrator: mi.Integrator,
    rendering: RenderingConfig,
    sensor: Union[int, mi.Sensor] = 0,
    formats: list[str] = None,
) -> list[mi.Bitmap]:
    if formats is None:
        formats = ["png", "exr"]

    with torch.no_grad():
        with dr.suspend_grad():
            img = mi.render(scene, spp=rendering.spp, integrator=integrator, sensor=sensor)

            if rendering.integrator.startswith("nerad"):
                _, LHS, RHS = process_nerad_output(img)
                save_image(folder / "rhs", name, formats, RHS)
                save_image(folder / "lhs", name, formats, LHS)
                return [LHS, RHS]
            else:
                save_image(folder, name, formats, img)
                return [img]
