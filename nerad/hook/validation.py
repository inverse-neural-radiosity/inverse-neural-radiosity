import logging
from pathlib import Path
from typing import Any

import mitsuba as mi
from omegaconf import OmegaConf

from common import create_integrator
from nerad.integrator.highquality import HighQuality
from nerad.model.config import RenderingConfig, ValidationConfig
from nerad.utils.dict_utils import inject_dict
from nerad.utils.render_utils import render_and_save_image

logger = logging.getLogger(__name__)


class ValidationHook:
    def __init__(self, cfg: ValidationConfig, train_render: RenderingConfig, scene: mi.Scene, integrator_injection: dict[str, Any], kwargs_injection: dict[str, Any]):
        self.cfg = cfg
        self.scene = scene
        self.use_hq = True

        # inject config from training
        train_render = OmegaConf.to_container(train_render)
        val_render = OmegaConf.to_container(cfg.rendering)
        inject_dict(val_render, train_render)
        logger.info(f"Integrator for validation [{cfg.name}]: {val_render}")

        self.rendering: RenderingConfig = OmegaConf.create(val_render)

        self.integrator = create_integrator(
            self.rendering, scene, {"samples_per_pass": 1}, integrator_injection, kwargs_injection)
        if (self.use_hq):
            block_size = 32
            self.integrator = HighQuality(block_size, self.integrator)
            logger.info(
                f"High quality renderer being used for integrator {self.integrator} at block size: {block_size}")

    def get_integrator(self):
        return self.integrator.integrator if self.use_hq else self.integrator

    def run(self, step: int, out_root: Path, name: str, sensor):
        cfg = self.cfg

        if step % cfg.step_size != 0 and not (cfg.first_step and step == 1):
            return None

        return render_and_save_image(
            out_root / "validation" / str(step) / cfg.name,
            name,
            self.scene,
            self.integrator,
            self.rendering,
            sensor,
        )
