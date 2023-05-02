import drjit as dr

from nerad.loss import LossFucntion, loss_registry


@loss_registry.register("l2")
class L2Loss(LossFucntion):
    def compute_loss(self, img, gt):
        return dr.sqr(img - gt)
