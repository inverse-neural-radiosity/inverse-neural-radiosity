import drjit as dr
import mitsuba as mi

from nerad.integrator import register_integrator


@register_integrator("albedo")
class AlbedoIntegrator(mi.SamplingIntegrator):
    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               medium: mi.Medium,
               active: mi.Bool):
        with dr.suspend_grad():
            # Standard BSDF evaluation context for path tracing
            bsdf_ctx = mi.BSDFContext()

            # --------------------- Configure loop state ----------------------

            # Copy input arguments to avoid mutating the caller's state
            ray = mi.Ray3f(dr.detach(ray))
            # Depth of current vertex
            depth = mi.UInt32(0)
            active = mi.Bool(active)                      # Active SIMD lanes

            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=dr.eq(depth, 0))

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)
            reflect = bsdf.eval_diffuse_reflectance(si)

        return (reflect, si.is_valid(), [])
