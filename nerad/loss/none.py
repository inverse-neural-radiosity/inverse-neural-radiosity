from nerad.loss import LossFucntion, loss_registry


@loss_registry.register("none")
class DeactiveLoss(LossFucntion):
    def compute_loss(self, img, gt):
        return gt*0
