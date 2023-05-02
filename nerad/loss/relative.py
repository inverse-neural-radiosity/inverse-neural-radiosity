import drjit as dr

from nerad.loss import LossFucntion, loss_registry


class RelativeLoss(LossFucntion):
    def __init__(self, n_steps: int) -> None:
        super().__init__()
        self.step = -1
        self.n_steps = n_steps
        self.exponent = 0

    def normalize_loss(self, denominator, numerator, annealing):
        normalizer = (dr.detach(denominator)+0.01)
        if (annealing):
            normalizer = normalizer**(self.exponent)
        normalized = (numerator/normalizer)
        return normalized

    def update_state(self, step: int):
        self.step = step
        self.exponent = max(0, 1 - self.step/(self.n_steps*0.8))

    def denominator(self, img, gt):
        raise NotImplementedError()


class L2RelativeLoss(RelativeLoss):
    def __init__(self, n_steps: int, annealing: bool) -> None:
        super().__init__(n_steps)
        self.annealing = annealing

    def compute_loss(self, img, gt):
        if self.annealing:
            assert self.step >= 0
        non_relative_loss = dr.sqr(img - gt)
        denom = self.denominator(img, gt)
        rel_loss = self.normalize_loss(denom, non_relative_loss, self.annealing)
        return rel_loss


@loss_registry.register("l2_relative_gt")
class L2RelativeGt(L2RelativeLoss):
    def denominator(self, img, gt):
        return dr.sqr(gt)


@loss_registry.register("l2_relative_prediction")
class L2RelativePrediction(L2RelativeLoss):
    def denominator(self, img, gt):
        return dr.sqr(img)


@loss_registry.register("l2_relative_both")
class L2RelativeBoth(L2RelativeLoss):
    def denominator(self, img, gt):
        return dr.sqr(0.5*(gt+img))
