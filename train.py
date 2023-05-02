import logging
import os
import time
from os.path import isfile
from pathlib import Path
from typing import Optional

import drjit as dr
import hydra
import mitsuba as mi
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter

from common import (compute_output_metrics, configure_compute,
                    create_integrator, create_loss_function, load_dataset,
                    prepare_learned_objects)
from mytorch.exp_lr import ExpLR, ExpLRScheduler
from mytorch.pbar import ProgressBar, ProgressBarConfig
from mytorch.utils.profiling_utils import (counter_profiler, time_profiler,
                                           vram_profiler)
from nerad.hook.save_checkpoint import SaveCheckpointHook
from nerad.hook.save_image import SaveImageHook
from nerad.hook.validation import ValidationHook
from nerad.model.config import TrainConfig
from nerad.utils.data_utils import create_index_loader
from nerad.utils.debug_utils import check_gradients
from nerad.utils.image_utils import save_image, find_nonzero_batches
from nerad.utils.json_utils import write_json
from nerad.utils.mitsuba_utils import swap_roughness_net_and_albedo_net
from nerad.utils.render_utils import process_nerad_output
from nerad.utils.sensor_utils import create_sensor

logger = logging.getLogger(__name__)
pbar_config = ProgressBarConfig(1, ["recons", "residual", "LHS_recons", "recons_coarse"], True)


@hydra.main(version_base="1.2", config_path="config", config_name="train")
def main(cfg: TrainConfig = None):
    print(OmegaConf.to_yaml(cfg))

    # profiling flags
    time_profiler.enabled = cfg.profile_time
    time_profiler.synchronize_cuda = cfg.profile_time_sync_cuda
    vram_profiler.enabled = cfg.profile_vram
    counter_profiler.enabled = cfg.profile_counter
    if cfg.disable_optimizer_step:
        logger.info("Optimizer step call is disabled")

    configure_compute(cfg.compute)
    device = os.environ.get("TORCH_DEVICE", "cuda:0")

    out_root = Path(HydraConfig.get().runtime.output_dir)
    logger.info(f"Output: {out_root}")

    if cfg.is_watchdog_init:
        print(f"watchdog:out_root:{out_root}")
        return

    scene, transforms, images, learned_modules, gt_albedo, gt_roughness = load_dataset(cfg.dataset, cfg.bsdf, device)
    n_views = len(images)
    # Assumption: dataset is squared
    resolution = images[0].width()

    rendering = cfg.rendering
    is_nerad = rendering.integrator.startswith("nerad")

    recons_loss_function = create_loss_function(cfg.recons_loss, cfg.n_steps)
    loss_functions = [recons_loss_function]
    logger.info(f"recons_loss_function: {recons_loss_function}")
    if is_nerad:
        LHS_recons_loss_function = create_loss_function(cfg.LHS_recons_loss, cfg.n_steps)
        loss_functions.append(LHS_recons_loss_function)
        logger.info(f"LHS_recons_loss_function: {LHS_recons_loss_function}")

    integrator_injection = {}
    if is_nerad:
        residual_loss_function = create_loss_function(cfg.residual_loss, cfg.n_steps)
        loss_functions.append(residual_loss_function)
        integrator_injection["residual_function"] = residual_loss_function

    integrator_function_injection = {"device": device}
    integrator = create_integrator(
        rendering,
        scene,
        post_init_injection=integrator_injection,
        kwargs_injection=integrator_function_injection,
    )

    learned_info = prepare_learned_objects(
        scene,
        integrator,
        learned_modules,
        cfg.envmap,
        cfg,
        find_latest_ckpt(out_root / "checkpoints") if cfg.resume else None,
        find_latest_ckpt(Path(cfg.radiance_cache) / "checkpoints") if cfg.radiance_cache is not None else None,
        device,
    )

    start_step: int = learned_info["step"] + 1
    params: mi.SceneParameters = learned_info["params"]
    integrator_params: mi.SceneParameters = learned_info["integrator_params"]
    torch_optim: torch.optim.Adam = learned_info["torch_optim"]
    mi_optim: mi.ad.Adam = learned_info["mi_optim"]
    torch_scheduler: ExpLRScheduler = learned_info["torch_scheduler"]
    mi_scheduler: ExpLR = learned_info["mi_scheduler"]
    learned_modules: dict[str, nn.Module] = learned_info["learned_modules"]
    mi_optimized_params = learned_info["mi_optimized_params"]

    end_step = cfg.n_steps
    if end_step <= start_step:
        logger.warning(f"end_step ({end_step}) <= start_step ({start_step}), aborting")
        return

    # Validation hooks
    validation_hooks = [ValidationHook(val_cfg, rendering, scene, integrator_injection, integrator_function_injection)
                        for val_cfg in cfg.validation.values()]
    for hook in validation_hooks:
        if is_nerad and hook.rendering.integrator.startswith("nerad"):
            hook.get_integrator().network = integrator.network

    # Hooks for saving mitsuba images
    image_hooks: list[SaveImageHook] = []
    if learned_info["learn_envmap"]:
        save_image(out_root / "gt_envmap", "envmap", ["png", "exr"], learned_info["gt_envmap"])
        image_hooks.append(
            SaveImageHook(
                cfg.envmap.save_step_size,
                cfg.envmap.save_first_step,
                "envmap",
                lambda: params[learned_info["envmap_key"]]
            )
        )

    # Saving hooks
    saving_hooks = [SaveCheckpointHook(save_cfg) for save_cfg in cfg.saving.values()]

    # Index loader
    loader = create_index_loader(n_views, cfg.shuffle)

    # Tensorboard
    writer = SummaryWriter(out_root / "tensorboard")

    # preprocess the offsets that do not result in a totally zero image (or with at least some pixels on)
    valid_offsets = {}
    if cfg.avoid_empty_batches:
        for idx, image in enumerate(images):
            img_tensor = mi.TensorXf(image).torch()
            valid_offsets[str(idx)] = find_nonzero_batches(img_tensor, cfg.batch_size)

    # Training loop
    time_profiler.start("training")
    logger.info(f"Training from step {start_step} to {end_step}")
    logger.info(f"Training started at {time.time()}")
    pbar = ProgressBar(pbar_config, end_step - start_step + 1)
    for step in range(start_step, end_step + 1):
        if torch_optim is not None:
            torch_optim.zero_grad()

        if not cfg.profile_time:
            torch.cuda.empty_cache()
            dr.flush_malloc_cache()

        vram_profiler.take_snapshot(f"{step}_start")

        for loss_function in loss_functions:
            loss_function.update_state(step - 1)

        view_idx = int(next(loader))
        sensor = create_sensor(
            resolution,
            transforms[str(view_idx)],
            random_crop=True,
            crop_size=cfg.batch_size,
            valid_offsets=valid_offsets.get(str(view_idx), None),
        )
        gt = images[view_idx]
        gt = crop_ground_truth(gt, sensor)

        time_profiler.start("forward")
        img = mi.render(
            scene,
            sensor=sensor,
            params=params,
            spp=rendering.spp,
            seed=step,
            seed_grad=step+1,
            integrator=integrator,
        )

        if is_nerad:
            residual, LHS, RHS = process_nerad_output(img)
            img = RHS
            residual = dr.mean_nested(residual)
        time_profiler.end("forward")

        # record loss values every several steps to reduce GPU-CPU comm
        loss_values = {}

        def record_loss_value(key, value):
            if step % cfg.update_loss_step_size != 0:
                return
            loss_values[key] = float(str(value[0]))

        time_profiler.start("loss")
        loss = dr.mean(recons_loss_function.compute_loss(img, gt))
        record_loss_value("recons", loss)

        if is_nerad:
            loss += residual
            record_loss_value("residual", residual)

            LHS_recon_loss = dr.mean(LHS_recons_loss_function.compute_loss(LHS, gt))
            loss += LHS_recon_loss
            record_loss_value("LHS_recons", LHS_recon_loss)

        time_profiler.end("loss")

        vram_profiler.take_snapshot(f"{step}_forward")

        time_profiler.start("backward")
        dr.backward(loss, dr.ADFlag.ClearEdges)

        if not cfg.disable_optimizer_step:
            if torch_optim is not None:
                torch_optim.step()
                if torch_scheduler is not None:
                    torch_scheduler.step()
                    writer.add_scalar("torch_learning_rate", torch_optim.param_groups[0]["lr"], global_step=step)

            if mi_optim is not None:
                mi_optim.step()
                params.update(mi_optim)
                integrator_params.update(mi_optim)
                if mi_scheduler is not None:
                    lr = mi_scheduler.get_lr_rate(step) * cfg.learning_rate
                    mi_optim.set_learning_rate(lr)
                    writer.add_scalar("mi_learning_rate", lr, global_step=step)

        time_profiler.end("backward")

        vram_profiler.take_snapshot(f"{step}_backward")

        if cfg.check_gradients:
            for name, obj in learned_modules.items():
                logger.info(f"Check gradients of {name}")
                check_gradients(list(obj.parameters()))

        # Save checkpoints
        for hook in saving_hooks:
            hook.run(step, out_root, torch_optim, learned_modules, torch_scheduler, mi_optim, mi_optimized_params)

        # Validation
        val_view_idx = cfg.validation_view
        val_gt = {
            "image": images[val_view_idx],
            "albedo": gt_albedo[val_view_idx] if gt_albedo is not None else None,
            "roughness": gt_roughness[val_view_idx] if gt_roughness is not None else None,
        }
        start_val = time.time()
        for hook in validation_hooks:
            if hook.rendering.integrator == "roughness":
                swap_roughness_net_and_albedo_net(params)
            val_sensor = create_sensor(
                resolution,
                transforms[str(val_view_idx)],
            )
            val_outputs = hook.run(step, out_root, f"{val_view_idx:03d}", val_sensor)
            if hook.rendering.integrator == "roughness":
                swap_roughness_net_and_albedo_net(params)

            if val_outputs is None:
                continue

            val_metrics = compute_output_metrics(hook.cfg.name, val_outputs, rendering.integrator, val_gt)
            for key, value in val_metrics.items():
                writer.add_scalar(f"metric/{key}", value, global_step=step)
            if len(val_metrics) > 0:
                logger.info(
                    "Validation: " + ", ".join((f"{k}={v:.3f}" for k, v in val_metrics.items()))
                )
        # Save extra images
        for hook in image_hooks:
            hook.run(step, out_root)

        # Record time if any hook is run
        validation_time = time.time()-start_val
        if validation_time > 0.1:
            writer.add_scalar("metric/val_time", validation_time, global_step=step)

        loss_values.update({"total": sum(loss_values.values())})
        pbar.update(loss_values)

        for key, value in loss_values.items():
            writer.add_scalar(f"loss/{key}", value, global_step=step)

        counter_profiler.new_group()

        if cfg.abort_step_size > 0 and step % cfg.abort_step_size == 0:
            logger.warning(f"Aborting training every {cfg.abort_step_size}")
            break

    time_profiler.end("training")
    logger.info(f"Training ended at {time.time()}")

    if cfg.profile_time:
        result = time_profiler.get_results_string()
        logger.info(f"Time profiling:\n{result}")
        with open(out_root / "time.txt", "w", encoding="utf-8") as f:
            f.write(result + "\n")

    if cfg.profile_vram:
        write_json(out_root / "vram.json", vram_profiler.snapshots)

    if cfg.profile_counter:
        write_json(out_root / "counter.json", counter_profiler.data)

    writer.flush()


def crop_ground_truth(gt,  sensor):
    offset, csize = sensor.film().crop_offset(), sensor.film().crop_size()
    gt = mi.TensorXf(gt)[offset[1]: offset[1] + csize[1],
                         offset[0]: offset[0] + csize[0]]
    return gt


def find_latest_ckpt(folder: Path) -> Optional[Path]:
    if isfile(folder / "latest.ckpt"):
        return folder / "latest.ckpt"

    files = list(folder.glob("*.ckpt"))
    if len(files) == 0:
        return None

    files = sorted(
        [[int(file.stem), file] for file in files],
        key=lambda a: a[0]
    )
    return files[-1][1]


if __name__ == "__main__":
    main()
