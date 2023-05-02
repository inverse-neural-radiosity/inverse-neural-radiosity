import drjit as dr

from nerad.mitsuba_wrapper import MitsubaTensorWrapper2D, wrapper_registry


@wrapper_registry.register("envmap_tensor")
class MitsubaEnvmapTensorWrapper(MitsubaTensorWrapper2D):
    def _post_process(self, output):
        return dr.clamp(output, 0, 10000)
