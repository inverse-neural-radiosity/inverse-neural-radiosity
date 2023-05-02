from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from mytorch.utils.string_utils import format_float


@dataclass
class ProgressBarConfig:
    update_frequency: int
    data_keys: list[str]
    show_first_step: bool


class ProgressBar:
    def __init__(self, config: ProgressBarConfig, total: int) -> None:
        self.config = config
        if config.update_frequency <= 0 or total <= 0:
            return

        self.total = total
        self.pbar = tqdm(total=total)
        self.steps = 0
        # key -> [sum, count]
        self.accumulated_data = defaultdict(lambda: [0, 0])

    def update(self, data: dict[str, Any]):
        if self.config.update_frequency <= 0:
            return

        self.steps += 1
        for key in self.config.data_keys:
            if key not in data:
                continue

            acc = self.accumulated_data[key]
            acc[0] += float(data[key])
            acc[1] += 1

        if self.steps == 1 and self.config.show_first_step or \
                self.steps % self.config.update_frequency == 0:

            averages = {k: v[0] / v[1] for k, v in self.accumulated_data.items()}

            self.pbar.update(self.steps - self.pbar.n)
            self.pbar.set_postfix_str(
                ", ".join([
                    f"{key}={format_float(averages[key])}" for key in self.config.data_keys if key in averages
                ])
            )
            self.accumulated_data.clear()
