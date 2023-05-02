import logging
from pathlib import Path
from typing import Any, Callable

from nerad.utils.render_utils import save_image

logger = logging.getLogger(__name__)


class SaveImageHook:
    def __init__(self, step_size: int, first_step: bool, name: str, getter: Callable[[], Any]):
        self.step_size = step_size
        self.first_step = first_step
        self.name = name
        self.getter = getter

    def run(self, step: int, out_root: Path):
        if step % self.step_size != 0 and not (self.first_step and step == 1):
            return

        logger.info(f"Save image {self.name}")
        save_image(
            out_root / "validation" / str(step) / "extra",
            self.name,
            ["png", "exr"],
            self.getter(),
        )
