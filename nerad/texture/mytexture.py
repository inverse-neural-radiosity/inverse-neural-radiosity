from typing import Any

import mitsuba as mi
import torch.nn as nn

from nerad.mitsuba_wrapper import wrapper_registry
from nerad.texture import register_texture


@register_texture("mytexture")
class MyTexture(mi.Texture, nn.Module):
    """An actual texture wrapper to hook SVBRDF parameters.
    When we load a Mitsuba scene XML we replace the texture definitions with this class to make them trainable.

    The 'network' is a MitsubaWrapper that can be an MLP or a dense grid.
    """
    def __init__(self, props: mi.Properties) -> None:
        mi.Texture.__init__(self, props)
        nn.Module.__init__(self)
        self.network = None

    def post_init(
        self,
        function: str,
        kwargs: dict[str, Any],
    ):
        self.network = wrapper_registry.build(function, kwargs)

    def traverse(self, callback):
        if self.network is not None:
            self.network.traverse(callback)
        callback.put_parameter("texture", self, mi.ParamFlags.NonDifferentiable)

    def eval(self, si, active=True, dirs=None, norms=None, albedo=None):
        return self.network.eval(si.p, dirs, norms, albedo)

    def eval_1(self, si, active=True):
        return mi.Float(self.eval(si)[0])

    def eval_1_grad(self, *args, **kwargs):
        raise NotImplementedError()

    def eval_3(self, *args, **kwargs):
        raise NotImplementedError()

    def mean(self, *args, **kwargs):
        raise NotImplementedError()

    def to_string(self):
        return (
            "MyTexture[\n"
            f"  network={self.network}\n"
            "]"
        )
