import re
from pathlib import Path

import drjit as dr
import mitsuba as mi


def vec_to_tens_safe(vec):
    # Converts a Vector3f to a TensorXf safely in mitsuba while keeping the gradients;
    # a regular type cast mi.TensorXf(vector) detaches the gradients
    return mi.TensorXf(dr.ravel(vec), shape=[dr.shape(vec)[1], dr.shape(vec)[0]])


def load_scene_with_custom_bsdf(file, mode: str) -> mi.Scene:
    """Edit XML and assign one single SVBRDF to all shapes in the scene

    See nerad/texture/mytexture.py
    """

    assert mode in scene_editing_bsdf

    if 'living-room-2' in file or 'veach_ajar' in file or 'nerf_scenes' in file or 'cube' in file or 'bunny' in file:
        file = Path(file)
        scene_txt = (file).read_text("utf-8")

        shapes_file = file.parent/'shapes.xml'
        shape_txt = (shapes_file).read_text("utf-8")

        # Use regex to assign our bsdf name to all shapes
        modified_shape_txt = re.sub('ref id=".*"', 'ref id="my-bsdf" name="bsdf"', shape_txt)

        # Insert our bsdf to the list of BSDFs
        modified_scene_txt = re.sub('<include filename="shapes.xml"/>',
                                    '<include filename="shapes.modified.xml"/>', scene_txt)
        modified_scene_txt = re.sub(
            f'<include filename="materials_{mode}.xml"/>', f'<include filename="materials_{mode}.xml"/>' + f"{scene_editing_bsdf[mode]}\n", modified_scene_txt)

        modified_scene_file = file.parent / (file.stem + ".modified.xml")
        modified_shape_file = shapes_file.parent / (shapes_file.stem + ".modified.xml")

        modified_scene_file.write_text(modified_scene_txt, "utf-8")
        modified_shape_file.write_text(modified_shape_txt, "utf-8")
        scene = mi.load_file(str(modified_scene_file))
    else:
        file = Path(file)
        data = file.read_text("utf-8")

        # Use regex to assign our bsdf name to all shapes
        modified_data = re.sub('ref id=".*"', 'ref id="my-bsdf"', data)
        # Insert our bsdf to the list of BSDFs
        modified_data = modified_data.replace("<shape", f"{scene_editing_bsdf[mode]}\n<shape", 1)

        modified_file = file.parent / (file.stem + ".modified.xml")
        modified_file.write_text(modified_data, "utf-8")
        scene = mi.load_file(str(modified_file))
        modified_file.unlink()
    return scene


builtin_bsdf_required_textures = {
    "diffuse": ["reflectance"],
    "principled": ["base_color", "roughness"],
}


scene_editing_bsdf = {
    "mydiffbsdf": """
    <bsdf type="twosided" id="my-bsdf">
        <bsdf type="mydiffbsdf">
        </bsdf>
    </bsdf>""",
    "diffuse": """
     <bsdf type="twosided" id="my-bsdf">
        <bsdf type="diffuse">
            <texture name="reflectance" type="mytexture">
            </texture>
        </bsdf>
    </bsdf>""",
    "principled": """
    <bsdf type="twosided" id="my-bsdf">
        <bsdf type="principled">
            <texture name="base_color" type="mytexture">
            </texture>
            <texture name="roughness" type="mytexture">
            </texture>
            <float name="metallic" value="$metallic" />
            <float name="specular" value="$specular" />
            <float name="spec_tint" value="$spec_tint" />
            <float name="anisotropic" value="$anisotropic" />
            <float name="sheen" value="$sheen" />
            <float name="sheen_tint" value="$sheen_tint" />
            <float name="clearcoat" value="$clearcoat" />
            <float name="clearcoat_gloss" value="$clearcoat_gloss" />
            <float name="spec_trans" value="$spec_trans" />
        </bsdf>
    </bsdf>""",
}


def load_scene_with_roughness_data(file) -> mi.Scene:
    """Edit XML and assign one single SVBRDF to all shapes in the scene"""

    if 'living-room-2' in file or 'veach_ajar' in file or 'nerf_scenes' in file or 'bunny' in file:
        file = Path(file)
        scene_txt = (file).read_text("utf-8")

        brdf_file = file.parent/'materials_principled.xml'
        brdf_txt = (brdf_file).read_text("utf-8")

        # Replace all roughness with everything
        brdf_txt = brdf_txt.replace('base_color', 'halalalaunqiue')
        brdf_txt = brdf_txt.replace('roughness', 'base_color')
        brdf_txt = brdf_txt.replace('halalalaunqiue', 'roughness')

        scene_txt = re.sub('<include filename="materials_principled.xml"/>',
                           '<include filename="materials_principled.modified_roughness.xml"/>', scene_txt)

        modified_scene_file = file.parent / (file.stem + ".modified_roughness.xml")
        modified_brdf_file = brdf_file.parent / (brdf_file.stem + ".modified_roughness.xml")

        modified_scene_file.write_text(scene_txt, "utf-8")
        modified_brdf_file.write_text(brdf_txt, "utf-8")

        scene = mi.load_file(str(modified_scene_file))
        modified_scene_file.unlink()
        modified_brdf_file.unlink()

    else:
        # left here for backward compatibility
        file = Path(file)
        data = file.read_text("utf-8")

        # Replace all roughness with everything
        data = data.replace('base_color', 'halalalaunqiue')
        data = data.replace('roughness', 'base_color')
        data = data.replace('halalalaunqiue', 'roughness')

        modified_file = file.parent / (file.stem + ".modified_roughness.xml")
        modified_file.write_text(data, "utf-8")
        scene = mi.load_file(str(modified_file))
        modified_file.unlink()
    return scene


def swap_roughness_net_and_albedo_net(params, using_gt_brdf=False, scene_file=None):
    """There is no easy way to render material roughness in Mitsuba.
    Therefore, we assign the roughness MLP to an albedo object and render the "albedo".

    See call sites for usage.
    """

    if not using_gt_brdf:
        reflectance_name = 'my-bsdf.brdf_0.base_color.texture'
        reflectance = params[reflectance_name]
        ref_params = mi.traverse(reflectance)

        # find learned (mitsuba_wrapper) key
        key = [k for k in ["network", "tensor", "mi_texture"] if k in ref_params][0]
        ref_net = ref_params[key]

        roughness_name = 'my-bsdf.brdf_0.roughness.texture'
        roughness = params[roughness_name]
        rough_params = mi.traverse(roughness)
        rough_net = rough_params[key]

        # see MitsubaWrapper definitions
        match key:
            case "network":
                roughness.network.network = ref_net
                reflectance.network.network = rough_net
            case "tensor":
                roughness.network.tensor = ref_net
                reflectance.network.tensor = rough_net
            case "mi_texture":
                roughness.network.texture = ref_net
                reflectance.network.texture = rough_net
    else:
        return load_scene_with_roughness_data(scene_file)


def get_batch_size(spp):
    """
    Get the maximum power of 2 batch size possible given the 2^30 limit by mitsuba for the wavefront size
    """
    maximum_wavefrontsize = 2**30
    return 2**int(dr.log2(maximum_wavefrontsize/spp)/2)
