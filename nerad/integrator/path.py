import drjit as dr
import mitsuba as mi
import numpy as np

from mytorch.utils.profiling_utils import counter_profiler
from nerad.integrator import register_integrator
from nerad.utils.render_utils import mis_weight


@register_integrator("mypath")
class MyPathTracer(mi.SamplingIntegrator):
    """Path Tracer (AD-PT)

    Code is tranlated from Mitsuba C++.
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self._init(**props.get("config").dict)

    def _init(
        self,
        hide_emitters: bool,
        return_depth: bool,
        max_depth: int,
        rr_depth: int,
        rr_prob: float = 0.95,
        **kwargs
    ):
        self.hide_emitters = hide_emitters
        self.return_depth = return_depth
        self.rr_prob = rr_prob

        # max depth
        if max_depth < 0 and max_depth != -1:
            raise Exception(
                "\"max_depth\" must be set to -1 (infinite) or a value >= 0")

        # Map -1 (infinity) to 2^32-1 bounces
        self.max_depth = max_depth if max_depth != -1 else 0xffffffff

        if rr_depth <= 0:
            raise Exception(
                "\"rr_depth\" must be set to a value greater than zero!")
        self.rr_depth = rr_depth

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

        # Record the following loop in its entirety
        loop = mi.Loop(name="MyPathTracer",
                       state=lambda: (sampler, ray, throughput, result,
                                      eta, depth, valid_ray, prev_si, prev_bsdf_pdf,
                                      prev_bsdf_delta, active))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)

        while loop(active):
            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.

            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=dr.eq(depth, 0))

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            em_hit_result = self.emitter_hit(
                scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si)
            result += em_hit_result
            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()

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
            # Don't run another iteration if the throughput has reached zero
            throughput_max = dr.max(throughput)
            rr_prob = dr.minimum(throughput_max * eta**2, self.rr_prob)
            rr_active = depth >= self.rr_depth
            rr_continue = sampler.next_1d() < rr_prob
            throughput[rr_active] *= dr.rcp(dr.detach(rr_prob))
            active = active_next & (
                ~rr_active | rr_continue) & dr.neq(throughput_max, 0)

        aov = [depth] if self.return_depth else []
        if counter_profiler.enabled:
            counter_profiler.record("integrator.depth", np.array(depth).tolist())

        return dr.select(valid_ray, result, 0), valid_ray, aov

    def aov_names(self):
        return ['depth'] if self.return_depth else []

    def bsdf_sample(self, sampler, active, bsdf_ctx, si, bsdf, active_next):

        bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                               sampler.next_1d(),
                                               sampler.next_2d(),
                                               active_next)

        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        # When the path tracer is differentiated, we must be careful that
        #   the generated Monte Carlo samples are detached (i.e. don't track
        #   derivatives) to avoid bias resulting from the combination of moving
        #   samples and discontinuous visibility. We need to re-evaluate the
        #   BSDF differentiably with the detached sample in that case. */
        if (dr.grad_enabled(ray)):
            ray = dr.detach(ray)

            # Recompute 'wo' to propagate derivatives to cosine term
            wo = si.to_local(ray.d)
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active)
            bsdf_weight[bsdf_pdf > 0] = bsdf_val / dr.detach(bsdf_pdf)

        return bsdf_sample, bsdf_weight, ray

    def emitter_hit(self, scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si):

        # Compute MIS weight for emitter sample from previous bounce
        ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

        mis = mis_weight(
            prev_bsdf_pdf,
            scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
        )

        em_hit_result = throughput * mis * ds.emitter.eval(si)
        return em_hit_result

    def sample_emitter(self, scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next):
        # Is emitter sampling even possible on the current vertex?
        active_em = active_next & mi.has_flag(
            bsdf.flags(), mi.BSDFFlags.Smooth)

        # If so, randomly sample an emitter without derivative tracking.
        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active_em)
        active_em &= dr.neq(ds.pdf, 0.0)

        if (dr.grad_enabled(si.p)):
            # Given the detached emitter sample, *recompute* its
            # contribution with AD to enable light source optimization
            ds.d = dr.normalize(ds.p - si.p)
            em_val = scene.eval_emitter_direction(si, ds, active_em)
            em_weight = dr.select(dr.neq(ds.pdf, 0), em_val / ds.pdf, 0)

            # Evaluate BSDF * cos(theta) differentiably
        wo = si.to_local(ds.d)
        bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
        em_sample_result = throughput * mis_em * bsdf_value_em * em_weight

        return em_sample_result

    def to_string(self):
        return (
            "MyPathTracer[\n"
            f"    max_depth={self.max_depth},\n"
            f"    rr_depth={self.rr_depth},\n"
            "]"
        )
