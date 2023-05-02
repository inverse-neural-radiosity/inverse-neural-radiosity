from typing import Any
import drjit as dr
import mitsuba as mi
import torch.nn as nn

from nerad.bsdf import register_bsdf
from nerad.mitsuba_wrapper import wrapper_registry


@register_bsdf("mydiffbsdf")
class MyDiffuseBSDF(mi.BSDF, nn.Module):
    """Custom Diffuse BSDF implementation copied from Mitsuba

    Note: this is unused in this project, kept for reference only
    """

    def __init__(self, props: mi.Properties):
        nn.Module.__init__(self)
        mi.BSDF.__init__(self, props)
        self.texture = None
        self.m_flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
        self.m_components = [self.m_flags]

    def post_init(
        self,
        function: str,
        kwargs: dict[str, Any],
    ):
        self.texture = wrapper_registry.build(function, kwargs)

    def sample(self, ctx, si, sample1, sample2, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        active &= cos_theta_i > 0

        bs = mi.BSDFSample3f()
        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = mi.BSDFFlags.DiffuseReflection
        bs.sampled_component = 0

        value = self.texture.eval((si.p))

        return (bs, dr.select(active & (bs.pdf > 0.0), value, mi.Vector3f(0)))

    def eval(self, ctx, si, wo, active):
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return mi.Vector3f(0)

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        value = self.texture.eval(si.p)
        value = value * dr.inv_pi * cos_theta_o

        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), value, mi.Vector3f(0))

    def eval_diffuse_reflectance(self, si, active=True):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        value = self.texture.eval(si.p)

        return dr.select((cos_theta_i > 0.0), value, mi.Vector3f(0))

    def eval_pdf(self, ctx, si, wo, active):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def pdf(self, ctx, si, wo, active):
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return mi.Vector3f(0)

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)

        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)

    def traverse(self, callback):
        if self.texture is not None:
            self.texture.traverse(callback)
        callback.put_parameter("bsdf", self, mi.ParamFlags.Differentiable)

    def to_string(self):
        return (
            "MyDiffuseBSDF[\n"
            f"    {self.texture}\n"
            "]"
        )
