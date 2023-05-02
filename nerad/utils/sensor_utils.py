import random
import re
from pathlib import Path

import drjit as dr
import mitsuba as mi


def create_transforms(scene: str, n_views: int):
    # Hardcoded transformations only valid for lego scene
    if 'nerf_scenes' in scene:
        fov = 40
        if 'dragon' in scene:
            fov = 70
        transforms = {}
        scene_object = mi.load_file(scene)
        center = 0.5 * (scene_object.bbox().min + scene_object.bbox().max)

        steps = int(dr.sqrt(n_views))
        for k in range(steps):
            for j in range(steps):
                i = k*steps + j
                radius = 4.2
                space = mi.scalar_rgb.warp.square_to_uniform_hemisphere
                h, w = j/steps, k/steps
                if 'dragon' in scene:
                    radius = 20
                    space = mi.scalar_rgb.warp.square_to_uniform_sphere
                    h, w = max(j/steps, 0.05), max(k/steps, 0.05)
                vec = space(mi.ScalarVector2f(h, w))
                temp = vec[1]
                vec[1] = vec[2]
                vec[2] = temp
                origin = vec*radius

                to_world = mi.ScalarTransform4f \
                    .look_at(target=center,
                             origin=origin,
                             up=[0, 1, 0])
                transforms[str(i)] = {
                    "to_world": to_world.matrix.numpy().tolist(),
                    "fov": fov,
                }
    elif 'living-room-2' in scene or 'staircase' in scene or 'kitchen' in scene or 'veach_ajar' in scene or 'cube' in scene or 'bunny' in scene:
        path = str(Path(scene).parent / "cameras.xml")
        sensors = sensors = mi.load_file(path).sensors()
        transforms = {}
        for i in range(len(sensors)):
            fov = int(re.findall(r"\d+", re.findall(r"x_fov = \[\d+\]", str(sensors[i]))[0])[0])
            transforms[str(i)] = {
                "to_world": mi.ScalarTransform4f(sensors[i].world_transform().matrix.numpy()).matrix.numpy().tolist(),
                "fov": fov,
            }

    elif 'myLivingRoom' in scene:
        path = str(Path(scene).parent / "cameras.xml")
        scene = mi.load_file(path)
        transforms = {}
        camerapose = scene.shapes()[-3].bbox().min
        right = scene.shapes()[-1].bbox().min
        lookat = scene.shapes()[-2].bbox().min
        lookat_start = lookat
        lookat_end = lookat
        end = camerapose
        start = right

        for i in range(n_views):
            current = (i)/n_views
            cam_pos = (end-start) * current + start
            lookat = (lookat_end-lookat_start)*current + lookat_start
            trans = mi.ScalarTransform4f.look_at(origin=cam_pos,
                                                 target=lookat,
                                                 up=[0, 1, 0])

            transforms[str(i)] = {
                'to_world': mi.ScalarTransform4f(trans).matrix.numpy().tolist(),
                'fov': 55,
            }
    return transforms


def create_sensor(resolution, transform, random_crop=False, crop_size=None, valid_offsets=None):
    return mi.load_dict(sensor_dict(resolution=resolution, fov=transform["fov"], to_world=transform["to_world"], random_crop=random_crop, crop_size=crop_size, valid_offsets=valid_offsets))


def sensor_dict(resolution, fov, to_world, random_crop, crop_size, valid_offsets):
    sensor = {
        "type": "perspective",
        "fov": fov,
        "to_world": mi.ScalarTransform4f(to_world),
        "film": {
                "type": "hdrfilm",
                "width": resolution,
                "height": resolution,
                "filter": {"type": "box"},
                "pixel_format": "rgba"
        }
        # TODO: All scene MUST be rgba in this scenario and use a box filter, even for ground truth
    }

    if random_crop:
        assert crop_size > 0
        assert (resolution-crop_size) >= 0
        if valid_offsets is not None and len(valid_offsets) > 0:
            crop_offset = random.choice(valid_offsets)
            crop_offset = [crop_offset[1].item(), crop_offset[0].item()]
        else:
            crop_offset = [
                random.randint(0, resolution-crop_size),
                random.randint(0, resolution-crop_size)
            ]

        sensor["film"]["crop_width"] = crop_size
        sensor["film"]["crop_height"] = crop_size
        sensor["film"]["crop_offset_x"] = crop_offset[0]
        sensor["film"]["crop_offset_y"] = crop_offset[1]

    return sensor
