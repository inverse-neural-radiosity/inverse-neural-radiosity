import drjit as dr

from nerad.mitsuba_wrapper import MitsubaTextureWrapper, wrapper_registry


@wrapper_registry.register("radiance_mitsuba")
class MitsubaRadianceTextureWrapper(MitsubaTextureWrapper):
    def _post_process(self, output):
        return dr.clamp(output, 0, 1)
