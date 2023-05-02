from torch.optim.lr_scheduler import _LRScheduler


class ExpLR:
    def __init__(
        self,
        decay_start: int,
        decay_rate: float,
        decay_steps: int,
        min_rate: float,
    ) -> None:
        self.decay_start = decay_start
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_rate = min_rate

    def get_lr_rate(self, step: int):
        if step < self.decay_start:
            return 1

        return max(
            self.min_rate,
            self.decay_rate ** ((step - self.decay_start) / self.decay_steps),
        )


class ExpLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        decay_start: int,
        decay_rate: float,
        decay_steps: int,
        min_rate: float,
        last_epoch=-1,
        verbose=False,
    ):
        self.compute = ExpLR(decay_start, decay_rate, decay_steps, min_rate)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        rate = self.compute.get_lr_rate(self.last_epoch)
        return [rate * base_lr for base_lr in self.base_lrs]
