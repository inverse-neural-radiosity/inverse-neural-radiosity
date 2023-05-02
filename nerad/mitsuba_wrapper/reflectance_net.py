from typing import Any

import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from nerad.mitsuba_wrapper import MitsubaWrapper, wrapper_registry
from nerad.model.tcnn_embedding import TcnnEmbedding
from nerad.utils.mitsuba_utils import vec_to_tens_safe


class ReflectanceMlp(nn.Module):
    def __init__(
        self,
        width: int,
        hidden: int,
        embedding: dict[str, Any],
    ):
        super().__init__()

        self.embedding = TcnnEmbedding(embedding)
        in_size = 3 + self.embedding.n_output_dims

        hidden_layers = []
        for _ in range(hidden):
            hidden_layers.append(nn.Linear(width, width))
            hidden_layers.append(nn.LeakyReLU(inplace=True))

        self.network = nn.Sequential(
            nn.Linear(in_size, width),
            nn.LeakyReLU(inplace=True),
            *hidden_layers,
            nn.Linear(width, 3),
            nn.Sigmoid()
        )

    def forward(self, points):
        net_in = torch.cat([points, self.embedding(points)], dim=-1)
        ret = self.network(net_in)

        return ret


@wrapper_registry.register("reflectance_net")
class MitsubaReflectanceNetworkWrapper(MitsubaWrapper):
    def __init__(
        self,
        width: int,
        hidden: int,
        embedding: dict[str, Any],
        scene_min: Any,
        scene_max: Any,
    ):
        super().__init__(scene_min, scene_max, "bsdf_net")
        self.network = ReflectanceMlp(width, hidden, embedding)

    def _eval(self, pts, dirs, norms, albedo):
        pts = 2 * pts - 1
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        torch_out = self.eval_torch(p_tensor)
        output = dr.unravel(mi.Vector3f, torch_out.array)
        return dr.clamp(output, 0, 1)

    @dr.wrap_ad(source="drjit", target="torch")
    def eval_torch(self, pts):
        return self.network(pts)

    def _traverse(self, callback):
        callback.put_parameter("network", self.network, mi.ParamFlags.Differentiable)


@wrapper_registry.register("fake_reflectance_net")
class FakeMitsubaReflectanceNetworkWrapper(MitsubaReflectanceNetworkWrapper):
    def __init__(
        self,
        width: int,
        hidden: int,
        embedding: dict[str, Any],
        scene_min: Any,
        scene_max: Any,
        value: float,
    ):
        super().__init__(width, hidden, embedding, scene_min, scene_max)
        self.value = value

    def _eval(self, pts, dirs, norms, albedo):
        output = super()._eval(pts, dirs, norms, albedo)
        return dr.clamp(self.value + 0.0001 * output, 0, 1)
