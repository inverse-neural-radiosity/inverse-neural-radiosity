import drjit as dr

from nerad.loss import LossFucntion, loss_registry


@loss_registry.register("l1")
class L1Loss(LossFucntion):
    def compute_loss(self, img, gt):
        return dr.abs(img - gt)
