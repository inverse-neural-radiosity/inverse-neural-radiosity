from typing import Any

import mitsuba as mi
import torch.nn as nn

from nerad.emitter import register_emitter
from nerad.mitsuba_wrapper import wrapper_registry


@register_emitter("myenvmap")
class MyEnvmap(mi.Emitter, nn.Module):
    def __init__(self, props: mi.Properties) -> None:
        mi.Emitter.__init__(self, props)
        nn.Module.__init__(self)
        self.network = None
        self.m_flags = mi.EmitterFlags.Infinite

    def post_init(
        self,
        function: str,
        kwargs: dict[str, Any],
    ):
        self.network = wrapper_registry.build(function, kwargs)

    def traverse(self, callback):
        if self.network is not None:
            self.network.traverse(callback)
        callback.put_parameter("my-envmap", self, mi.ParamFlags.NonDifferentiable)

    def eval(self, si: mi.SurfaceInteraction3f, active: bool = True) -> mi.Color3f:
        # TODO: implement this
        raise NotImplementedError()

    def sample_direction(self, it, sample: mi.Point2f, active: bool = True):
        # TODO: implement this
        raise NotImplementedError()

    def pdf_direction(self, it: mi.Interaction3f, ds: mi.DirectionSample3f, active: bool = True) -> float:
        # TODO: implement this
        raise NotImplementedError()

    def eval_direction(self, it, ds, active: bool = True) -> mi.Color3f:
        # TODO: implement this
        raise NotImplementedError()

    def sample_ray(self, time: float, sample1: float, sample2: mi.Point2f, sample3: mi.Point2f, active: bool = True) -> tuple[mi.Ray3f, mi.Color3f]:
        # We probably don't need this
        raise NotImplementedError()

    def bbox(self) -> mi.BoundingBox3f:
        return mi.BoundingBox3f()

    def to_string(self):
        return (
            "MyEnvmap[\n"
            f"    {self.network}\n"
            "]"
        )
