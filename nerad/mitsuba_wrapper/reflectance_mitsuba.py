import drjit as dr

from nerad.mitsuba_wrapper import MitsubaTextureWrapper, wrapper_registry


@wrapper_registry.register("reflectance_mitsuba")
class MitsubaReflectanceTextureWrapper(MitsubaTextureWrapper):
    def _post_process(self, output):
        return dr.clamp(output, 0, 1)
