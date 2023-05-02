import drjit as dr
import mitsuba as mi

from nerad.integrator import register_integrator


@register_integrator("roughness")
class RoughnessIntegrator(mi.SamplingIntegrator):
    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               medium: mi.Medium,
               active: mi.Bool):
        with dr.suspend_grad():

            ray = mi.Ray3f(dr.detach(ray))
            depth = mi.UInt32(0)
            active = mi.Bool(active)

            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=dr.eq(depth, 0))

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)
            reflect = mi.Float(bsdf.eval_diffuse_reflectance(si)[0])

        return (mi.Vector3f(reflect), si.is_valid(), [])
