from mytorch.registry import Registry, import_children


class LossFucntion:
    def compute_loss(self, img, gt):
        raise NotImplementedError()

    def update_state(self, step: int):
        pass


loss_registry = Registry("loss", LossFucntion)
import_children(__file__, __name__)
