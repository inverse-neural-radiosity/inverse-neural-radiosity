import logging
from os.path import isdir
from pathlib import Path

import hydra
import mitsuba as mi
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm import tqdm

from common import configure_compute, create_integrator
from nerad.integrator.highquality import HighQuality
from nerad.model.config import GenerateDataConfig
from nerad.utils.dict_utils import inject_dict
from nerad.utils.json_utils import read_json, write_json
from nerad.utils.mitsuba_utils import (get_batch_size,
                                       swap_roughness_net_and_albedo_net)
from nerad.utils.render_utils import render_and_save_image
from nerad.utils.sensor_utils import create_sensor, create_transforms

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config", config_name="generate_data")
def main(cfg: GenerateDataConfig = None):
    print(OmegaConf.to_yaml(cfg))
    hq = True

    configure_compute(cfg.compute)

    out_root = Path(HydraConfig.get().runtime.output_dir)
    out_dirs = [out_root / "png", out_root / "exr"]

    if any(isdir(d) for d in out_dirs) and not cfg.overwrite:
        logger.info(f"Some image output directories {out_dirs} exists, quit")
        logger.info(f"To delete them, run:\nrm -rf {out_root}")
        return

    rendering = cfg.rendering

    scene = mi.load_file(cfg.dataset.scene)
    if cfg.dataset.cameras is None:
        transforms = create_transforms(cfg.dataset.scene, cfg.dataset.n_views)
    else:
        transforms = read_json(cfg.dataset.cameras)
    write_json(out_root / "transforms.json", transforms)

    for val_cfg in cfg.validation.values():
        # inject config from training
        train_render = OmegaConf.to_container(cfg.rendering)
        val_render = OmegaConf.to_container(val_cfg.rendering)
        inject_dict(val_render, train_render)
        logger.info(f"Integrator for gt generation [{val_cfg.name}]: {val_render}")

        rendering = OmegaConf.create(val_render)

        if val_cfg.rendering.integrator == 'roughness':
            old_scene_obj = scene
            scene = swap_roughness_net_and_albedo_net(None, True, cfg.dataset.scene)

        integrator = create_integrator(rendering, scene)
        if (hq):
            block_size = get_batch_size(rendering.spp)
            integrator = HighQuality(block_size, integrator)
            logger.info(f"High quality renderer being used at block size: {block_size}")

        for idx in tqdm(range(len(transforms))):
            sensor = create_sensor(rendering.width, transforms[str(idx)])
            render_and_save_image(
                out_root / val_cfg.name if val_cfg.name != 'image' else out_root,
                f"{idx:03d}",
                scene,
                integrator,
                rendering,
                sensor,
            )

        if val_cfg.rendering.integrator == 'roughness':
            scene = old_scene_obj


if __name__ == "__main__":
    main()
