import drjit as dr

from nerad.mitsuba_wrapper import MitsubaTensorWrapper, wrapper_registry


@wrapper_registry.register("radiance_tensor")
class MitsubaRadianceTensorWrapper(MitsubaTensorWrapper):
    def _post_process(self, output):
        return dr.abs(output)
