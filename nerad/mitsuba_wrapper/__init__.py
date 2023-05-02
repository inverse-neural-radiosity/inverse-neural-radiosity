import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from mytorch.registry import Registry, import_children
from mytorch.utils.profiling_utils import counter_profiler, time_profiler
from nerad.utils.mitsuba_utils import vec_to_tens_safe


class MitsubaWrapper(nn.Module):
    """Wraps a learned object either from PyTorch or Mitsuba for querying in integrators

    Signature of eval() is specific to this project.
    """

    def __init__(self, scene_min: float, scene_max: float, name: str = None):
        super().__init__()
        self.grad_activator = mi.Vector3f(0)
        self.scene_min = scene_min
        self.scene_max = scene_max
        self.name = name or type(self).__name__

    def eval(self, pts, dirs=None, norms=None, albedo=None):
        if counter_profiler.enabled:
            counter_profiler.record(f"{self.name}.eval.pts", dr.shape(pts)[1])
        time_profiler.start(f"{self.name}.eval")
        pts = (pts - self.scene_min) / (self.scene_max - self.scene_min)
        result = self._eval(pts, dirs, norms, albedo)
        time_profiler.end(f"{self.name}.eval")
        return result

    def traverse(self, callback):
        callback.put_parameter("grad_activator", self.grad_activator, mi.ParamFlags.Differentiable)
        self._traverse(callback)

    def _eval(self, pts, dirs, norms, albedo):
        raise NotImplementedError()

    def _traverse(self, callback):
        pass


class MitsubaTensorWrapper(MitsubaWrapper):
    """Wraps a PyTorch dense grid"""

    def __init__(
        self,
        scene_min: float,
        scene_max: float,
        grid_size: int,
        value: float = 0.5,
    ):
        super().__init__(scene_min, scene_max)
        self.tensor = nn.parameter.Parameter(
            torch.ones(1, 3, grid_size, grid_size, grid_size) * value,
            requires_grad=True,
        )

    def _eval(self, pts, dirs, norms, albedo):
        pts = 2 * pts - 1
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        torch_out = self.eval_torch(p_tensor)
        output = dr.unravel(mi.Vector3f, torch_out.array)
        return self._post_process(output)

    def _post_process(self, output):
        raise NotImplementedError()

    @dr.wrap_ad(source="drjit", target="torch")
    def eval_torch(self, pts):
        return torch.nn.functional.grid_sample(
            self.tensor,
            pts[None, None, None],
            align_corners=False,
            padding_mode="border",
        ).view(3, -1).transpose(0, 1)

    def _traverse(self, callback):
        callback.put_parameter("tensor", self.tensor, mi.ParamFlags.Differentiable)


class MitsubaTensorWrapper2D(MitsubaWrapper):
    """Wraps a PyTorch 2D tensor"""

    def __init__(
        self,
        scene_min: float,
        scene_max: float,
        width: int,
        height: int,
        value: float = 0.5,
    ):
        super().__init__(scene_min, scene_max)
        self.tensor = nn.parameter.Parameter(
            torch.ones(1, 3, height, width) * value,
            requires_grad=True,
        )

    def _eval(self, pts, dirs, norms, albedo):
        pts = 2 * pts - 1
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        torch_out = self.eval_torch(p_tensor)
        output = dr.unravel(mi.Vector3f, torch_out.array)
        return self._post_process(output)

    def _post_process(self, output):
        raise NotImplementedError()

    @dr.wrap_ad(source="drjit", target="torch")
    def eval_torch(self, pts):
        return torch.nn.functional.grid_sample(
            self.tensor,
            pts[None, None],
            align_corners=False,
            padding_mode="border",
        ).view(3, -1).transpose(0, 1)

    def _traverse(self, callback):
        callback.put_parameter("tensor", self.tensor, mi.ParamFlags.Differentiable)


class MitsubaTextureWrapper(MitsubaWrapper):
    """Wraps a Mitsuba dense grid"""

    def __init__(
        self,
        scene_min: float,
        scene_max: float,
        grid_size: int,
        device: str,
        value: float = 0.5,
    ):
        super().__init__(scene_min, scene_max)
        value = torch.ones(grid_size, grid_size, grid_size, 3, device=device) * value
        self.texture = mi.Texture3f(mi.TensorXf(value), use_accel=False)

    def _eval(self, pts, dirs, norms, albedo):
        output = mi.Vector3f(self.texture.eval(pts))
        return self._post_process(output)

    def _post_process(self, output):
        raise NotImplementedError()

    def _traverse(self, callback):
        callback.put_parameter("mi_texture", self.texture.tensor(), mi.ParamFlags.Differentiable)


wrapper_registry = Registry("wrapper", MitsubaWrapper)
import_children(__file__, __name__)
