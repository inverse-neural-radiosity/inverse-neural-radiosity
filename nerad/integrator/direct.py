import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from nerad.integrator import register_integrator

from .path import MyPathTracer


@register_integrator("mydirect")
class MyDirectIntegrator(MyPathTracer, nn.Module):
    def __init__(self, props: mi.Properties):
        nn.Module.__init__(self)
        MyPathTracer.__init__(self, props)

    def get_albedo_detached(self, si):
        with dr.suspend_grad():
            with torch.no_grad():
                reflect = si.bsdf().eval_diffuse_reflectance(si)
        return reflect

    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               medium: mi.Medium,
               active: mi.Bool):

        depth = mi.UInt32(0)
        eta = mi.Float(1)
        # result = mi.Spectrum(0)
        throughput = mi.Spectrum(1)
        valid_ray = mi.Mask((~mi.Bool(self.hide_emitters))
                            & dr.neq(scene.environment(), None))

        active = mi.Bool(active)                      # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        bsdf_ctx = mi.BSDFContext()

        if isinstance(ray, mi.SurfaceInteraction3f):
            si = ray
            bsdf = si.bsdf()

        else:
            ray = mi.Ray3f(dr.detach(ray))
            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=dr.eq(depth, 0))
            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

        # ---------------------- Direct emission ----------------------

        E = self.emitter_hit(scene, throughput, prev_si,
                             prev_bsdf_pdf, prev_bsdf_delta, si)

        # ---------------------- Emitter sampling ----------------------

        active_next = si.is_valid()

        em_sample_result = self.sample_emitter(
            scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next)

        # ------------------ Detached BSDF sampling -------------------

        bsdf_sample, bsdf_weight, ray = self.bsdf_sample(
            sampler, active, bsdf_ctx, si, bsdf, active_next)

        # ------ Update loop variables based on current interaction ------

        throughput *= bsdf_weight
        eta *= bsdf_sample.eta
        valid_ray |= active & si.is_valid() & ~mi.has_flag(
            bsdf_sample.sampled_type, mi.BSDFFlags.Null)

        # Information about the current vertex needed by the next iteration
        prev_si = si
        prev_bsdf_pdf = bsdf_sample.pdf
        prev_bsdf_delta = mi.has_flag(
            bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

        # -------------------- Stopping criterion ---------------------

        depth[si.is_valid()] += 1
        # Don't run another iteration if the throughput has reached zero
        active = active_next

        si = scene.ray_intersect(ray,
                                 ray_flags=mi.RayFlags.All,
                                 coherent=dr.eq(depth, 0))

        # Get the BSDF, potentially computes texture-space differentials
        bsdf = si.bsdf(ray)

        # ---------------------- Direct emission ----------------------

        bsdf_sample_result = self.emitter_hit(
            scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si)

        RHS = bsdf_sample_result + em_sample_result
        rgb = dr.select(valid_ray, E + RHS, 0)

        return rgb, valid_ray, []

    def aov_names(self):
        return []

    def to_string(self):
        return (
            "MyDirectIntegrator[\n"
            "]"
        )

    def traverse(self, callback):
        pass
