import drjit as dr

from nerad.mitsuba_wrapper import MitsubaTensorWrapper, wrapper_registry


@wrapper_registry.register("reflectance_tensor")
class MitsubaReflectanceTensorWrapper(MitsubaTensorWrapper):
    def _post_process(self, output):
        return dr.clamp(output, 0, 1)
