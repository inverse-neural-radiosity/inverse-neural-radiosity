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


@register_integrator("nerad")
class Nerad(MyPathTracer, nn.Module):
    def __init__(self, props: mi.Properties):
        nn.Module.__init__(self)
        MyPathTracer.__init__(self, props)
        self.residual_sampler = None
        self.residual_function = None
        self.network = None
        # If a second residual needs to be computed, compute a second bounce
        # to compute the residual at the second bounce residual and ADD it to the primary residual
        self.compute_second_residual = props.get("config").dict.get("second_residual")
        # If a second (extra) residual needs to be computed (flag above is True), and flag below is Ture,
        # use random directions to compute the extra residual and REPLACE the primary residual with it
        # In essence this computes the residual the the primary hitpoint,
        # and not on the second bounce (despite it working only if self.compute_second_residual = True)
        # TODO: design here better
        self.compute_random_residual = props.get("config").dict.get("random_residual")
        self.detach_radiance_gradients_in_residual = props.get(
            "config").dict.get("detach_radiance_gradients_in_residual")
        self.detach_non_radiance_gradients_in_residual = props.get(
            "config").dict.get("detach_non_radiance_gradients_in_residual")
        self.detach_radiance_gradients_in_rhs = props.get("config").dict.get("detach_radiance_gradients_in_rhs")
        self.return_only_LHS = props.get("config").dict.get("return_only_LHS")

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

        compute_second_residual = self.compute_second_residual and medium is None

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
            compute_second_residual = False

        else:
            ray = mi.Ray3f(dr.detach(ray))
            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=dr.eq(depth, 0))
            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

        primary_si = si

        # ---------------------- Eval LHS ----------------------
        LHS = self.network.eval(
            si.p, si.to_world(si.wi), si.sh_frame.n, dr.detach(self.get_albedo_detached(si)))
        LHS = dr.select(active & si.is_valid(), LHS, mi.Vector3f(0))

        # ---------------------- Direct emission ----------------------

        E = self.emitter_hit(scene, throughput, prev_si,
                             prev_bsdf_pdf, prev_bsdf_delta, si)

        if self.return_only_LHS:
            mask = valid_ray | (active & si.is_valid())
            LHS = dr.select(mask, E + LHS, 0)
            zero_vec = LHS*0
            return zero_vec, mask, [LHS.x, LHS.y, LHS.z, dr.select(mask, mi.Float(1), mi.Float(0)), zero_vec.x, zero_vec.y, zero_vec.z]

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

        # ---------------------- Eval RHS ----------------------
        RHS_eval_doesnt_need_grad = self.detach_radiance_gradients_in_residual and self.detach_radiance_gradients_in_rhs
        with dr.suspend_grad(when=RHS_eval_doesnt_need_grad):
            with torch.set_grad_enabled(not RHS_eval_doesnt_need_grad):
                RHS_net = dr.select(active & si.is_valid(), self.network.eval(
                    si.p, -ray.d, si.sh_frame.n, dr.detach(self.get_albedo_detached(si))), mi.Vector3f(0))

        stop_grad_radiance = dr.detach if self.detach_radiance_gradients_in_rhs else lambda x: x
        RHS = stop_grad_radiance(RHS_net) * throughput + bsdf_sample_result + em_sample_result

        stop_grad_radiance = dr.detach if self.detach_radiance_gradients_in_residual else lambda x: x
        stop_grad_non_radiance = dr.detach if self.detach_non_radiance_gradients_in_residual else lambda x: x
        RHS_for_residual = stop_grad_radiance(RHS_net) * stop_grad_non_radiance(throughput) + \
            stop_grad_non_radiance(bsdf_sample_result + em_sample_result)

        aov = dr.select(valid_ray, E + LHS, 0)
        rgb = dr.select(valid_ray, E + RHS, 0)

        residual = self.residual_function.compute_loss(LHS, RHS_for_residual)
        residual = dr.select(primary_si.is_valid(), residual, mi.Vector3f(0))

        # TODO: second bounce residual computation, since there is no clean way to input this to this function, we would need to do this using medium, for now!
        if compute_second_residual:
            if (self.compute_random_residual):
                si = primary_si
                dir_samples = mi.warp.square_to_uniform_sphere(sampler.next_2d())
                si.wi = dir_samples
                active = True
            _, _, sec_bounce = self.sample(
                scene, sampler, si, 1, active & si.is_valid())
            sec_bounce_res = mi.Spectrum(
                sec_bounce[-3], sec_bounce[-2], sec_bounce[-1])
            if self.compute_random_residual:
                residual = sec_bounce_res
            else:
                residual += sec_bounce_res

        return rgb, valid_ray, [aov.x, aov.y, aov.z, dr.select(valid_ray, mi.Float(1), mi.Float(0)), residual.x, residual.y, residual.z]

    def aov_names(self):
        # warning: The below list must be in accordance with method process_nerad_output() and the outputs of the method sample() in this class
        return ["LHS.R", "LHS.G", "LHS.B", "LHS.a", "residual.x", "residual.y", "residual.z"]

    def to_string(self):
        return (
            "NeradIntegrator[\n"
            f"  network={self.network}\n"
            f"  residual_function={self.residual_function}\n"
            "]"
        )

    def traverse(self, callback):
        self.network.traverse(callback)
