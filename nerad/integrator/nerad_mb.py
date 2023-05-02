import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from nerad.integrator import register_integrator
from nerad.loss import LossFucntion
from nerad.mitsuba_wrapper import wrapper_registry
from nerad.model.sampler import ShapeSampler
from nerad.texture.dictionary import MiDictionary

from .path import MyPathTracer


@register_integrator("nerad_mb")
class NeradMB(MyPathTracer, nn.Module):
    '''
        Nerad multi bounce
        This integrator is a path tracer that enables neural radiosity as well.
        This integrator does not implement all the features of 'nerad.py',
        but it can do multi-bounce 'differentiable rendering'
        using neural radiosity, which is different than second bounce residual computation in 'nerad.py'. This class is not tested intensively and is prone to bugs,
        and so far was only used to create the bias figure in the paper.

    '''

    def __init__(self, props: mi.Properties):
        nn.Module.__init__(self)
        MyPathTracer.__init__(self, props)
        self.residual_sampler = None
        self.residual_function = None
        self.network = None

    def post_init(
        self,
        residual_function: LossFucntion,
        function: str,
        kwargs: MiDictionary,
    ):
        self.residual_function = residual_function
        self.network = wrapper_registry.build(function, kwargs)

    def get_albedo_detached(self, si):
        with dr.suspend_grad():
            with torch.no_grad():
                reflect = si.bsdf().eval_diffuse_reflectance(si)
        return reflect

    def compute_residual(self, scene, n, seed):
        if self.residual_sampler is None:
            self.residual_sampler = ShapeSampler(scene)

        si, prob = self.residual_sampler.sample_input(scene=scene, n=n, seed=seed)
        _, _, aov = self.sample(scene, self.residual_sampler.sampler, si, 1, True)
        residual = mi.Color3f(aov[-3:])
        return residual

    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               medium: mi.Medium,
               active: mi.Bool):

        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)
        eta = mi.Float(1)
        result = mi.Spectrum(0)
        throughput = mi.Spectrum(1)
        valid_ray = mi.Mask((~mi.Bool(self.hide_emitters))
                            & dr.neq(scene.environment(), None))

        active = mi.Bool(active)                      # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        bsdf_ctx = mi.BSDFContext()
        LHS = mi.Spectrum(0)
        d = 0

        while dr.any(active):
            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.

            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=dr.eq(depth, 0))

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Eval net ----------------------
            with dr.suspend_grad(when=d != 0):
                with torch.set_grad_enabled(d == 0):
                    network_contrib = self.network.eval(
                        si.p, si.to_world(si.wi), si.sh_frame.n, dr.detach(self.get_albedo_detached(si)))
                    network_contrib = dr.select(active & si.is_valid(), network_contrib, mi.Vector3f(0))

            # ---------------------- Direct emission ----------------------

            em_hit_result = self.emitter_hit(
                scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si)
            result += em_hit_result
            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            # ---------------------- net cont ----------------------
            result += dr.select(active & ~active_next, throughput*network_contrib, mi.Vector3f(0))
            if d == 0:
                LHS = dr.select(active & active_next, network_contrib + em_hit_result, mi.Vector3f(0))

            em_sample_result = self.sample_emitter(
                scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next)
            result += em_sample_result

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
            d += 1
            # Don't run another iteration if the throughput has reached zero
            throughput_max = dr.max(throughput)
            rr_prob = dr.minimum(throughput_max * eta**2, .95)
            rr_active = depth >= self.rr_depth
            rr_continue = sampler.next_1d() < rr_prob
            throughput[rr_active] *= dr.rcp(dr.detach(rr_prob))
            active = active_next & (
                ~rr_active | rr_continue) & dr.neq(throughput_max, 0)

        aov = dr.select(valid_ray,  LHS, 0)
        rgb = dr.select(valid_ray, result, 0)
        residual = self.residual_function.compute_loss(LHS, dr.detach(result))

        return rgb, valid_ray, [aov.x, aov.y, aov.z, dr.select(valid_ray, mi.Float(1), mi.Float(0)), residual.x, residual.y, residual.z]

    def aov_names(self):
        # warning: The below list must be in accordance with method process_nerad_output() and the outputs of the method sample() in this class
        return ["LHS.R", "LHS.G", "LHS.B", "LHS.a", "residual.x", "residual.y", "residual.z"]

    def to_string(self):
        return (
            "NeradMBIntegrator[\n"
            f"  network={self.network}\n"
            f"  residual_function={self.residual_function}\n"
            "]"
        )

    def traverse(self, callback):
        self.network.traverse(callback)
