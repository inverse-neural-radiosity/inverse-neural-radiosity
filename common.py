import logging
import random
from pathlib import Path
from typing import Any, Optional, Union

import drjit as dr
import mitsuba as mi
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from mytorch.exp_lr import ExpLR, ExpLRScheduler

from nerad.bsdf import registered_bsdfs
from nerad.integrator import registered_integrators
from nerad.loss import loss_registry
from nerad.model.config import (BsdfConfig, ComputeConfig, DatasetConfig, EnvmapConfig,
                                ObjectConfig, RenderingConfig, TrainConfig)
from nerad.texture import registered_textures
from nerad.utils.dict_utils import inject_dict
from nerad.utils.io_utils import glob_sorted
from nerad.utils.json_utils import read_json
from nerad.utils.metric_utils import compute_metrics
from nerad.utils.mitsuba_utils import (builtin_bsdf_required_textures,
                                       load_scene_with_custom_bsdf)

logger = logging.getLogger(__name__)


def configure_compute(cfg: ComputeConfig):
    logger.info(f"Set drjit flags to {cfg.dr_optimization_flags}")
    dr.set_flag(dr.JitFlag.LoopRecord, cfg.dr_optimization_flags)
    dr.set_flag(dr.JitFlag.VCallRecord, cfg.dr_optimization_flags)
    dr.set_flag(dr.JitFlag.VCallOptimize, cfg.dr_optimization_flags)

    logger.info(f"Set torch detech anomaly to {cfg.torch_detect_anomaly}")
    torch.autograd.set_detect_anomaly(cfg.torch_detect_anomaly)

    logger.info(f"Seed everything with {cfg.seed}")
    seed_everything(cfg.seed)

    log_mitsuba_registration()


def create_integrator(
    cfg: RenderingConfig,
    scene: mi.Scene,
    extra_config: dict[str, Any] = None,
    post_init_injection: dict[str, Any] = None,
    kwargs_injection: dict[str, Any] = None,
):
    mi_dict = {
        "type": cfg.integrator,
    }
    if extra_config is not None:
        mi_dict.update(extra_config)

    integrator_config = OmegaConf.to_container(cfg.config)
    if len(integrator_config) > 0:
        if cfg.integrator in registered_integrators:
            mi_dict["config"] = {
                "type": "dict",
                **integrator_config,
            }
        else:
            mi_dict.update(integrator_config)

    logger.info(f"Integrator dict: {mi_dict}")
    integrator = mi.load_dict(mi_dict)

    _mitsuba_post_init(cfg.post_init, integrator, scene, post_init_injection, kwargs_injection)
    logger.info(f"Integrator: {integrator}")
    return integrator


def load_dataset(cfg: DatasetConfig, bsdf_cfg: BsdfConfig, device: str):
    bsdf_name = bsdf_cfg.name
    texture_cfg = bsdf_cfg.texture
    learned_modules: dict[str, nn.Module] = {}

    if bsdf_name != "gt":
        # two cases: (1) built-in bsdf with custom texture, (2) custom bsdf
        is_custom_bsdf = False

        # for (1), check if custom textures presents
        required_textures = builtin_bsdf_required_textures.get(bsdf_name)
        if required_textures is not None:
            required_textures = sorted(required_textures)
            provided_textures = sorted(texture_cfg.keys())
            assert required_textures == provided_textures, \
                f"BSDF '{bsdf_cfg.name}' requires {required_textures} but got {provided_textures}"
        # for (2), check if registered
        else:
            assert bsdf_name in registered_bsdfs, f"BSDF '{bsdf_name}' is neither built-in nor custom"
            assert len(texture_cfg) == 0, "Custom BSDF does not support custom texture"
            is_custom_bsdf = True

        scene = load_scene_with_custom_bsdf(cfg.scene, bsdf_cfg.name)
        params = mi.traverse(scene)
        for key in params.keys():
            if not key.startswith("my-bsdf.") or not key.endswith(".texture"):
                continue

            if is_custom_bsdf:
                name = "bsdf"
                post_init_cfg = bsdf_cfg.post_init
            else:
                # key looks like: my-bsdf.brdf_0.reflectance.texture
                # we use the name "reflectance"
                name = key.split(".")[-2]
                post_init_cfg = texture_cfg[name].post_init

            obj = params.get(key)
            assert isinstance(obj, nn.Module)
            _mitsuba_post_init(post_init_cfg, obj, scene, kwargs_injection={"device": device})

            learned_modules[name] = obj
    else:
        scene = mi.load_file(cfg.scene)

    transforms = read_json(cfg.cameras)
    n_views = min(cfg.n_views, len(transforms)) if cfg.n_views > 0 else len(transforms)

    logger.info(f"Load {n_views} views from {len(transforms)} views")

    images = load_exr_files(Path(cfg.cameras).parent / "exr", n_views)
    assert len(images) == n_views

    # Handle old training
    cfg = OmegaConf.to_container(cfg)
    albedo_path = cfg.get("albedo")
    roughness_path = cfg.get("roughness")

    gt_albedo = None
    if albedo_path is not None:
        gt_albedo = load_exr_files(albedo_path, n_views)
        assert len(gt_albedo) == n_views

    gt_roughness = None
    if roughness_path is not None:
        gt_roughness = load_exr_files(roughness_path, n_views)
        assert len(gt_roughness) == n_views

    return scene, transforms, images, learned_modules, gt_albedo, gt_roughness


def load_exr_files(folder: Path, limit: int = 0):
    files = glob_sorted(folder, "*.exr")
    logger.info(f"Loading files from {folder}:\n" + ", ".join([f.name for f in files]))

    if limit <= 0:
        limit = len(files)

    return [
        mi.Bitmap(str(file)) for file in files[:limit]
    ]


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_mitsuba_registration():
    logger.info(f"Registered integrators: {' '.join(registered_integrators)}")
    logger.info(f"Registered bsdf: {' '.join(registered_bsdfs)}")
    logger.info(f"Registered texture: {' '.join(registered_textures)}")


def create_loss_function(config: ObjectConfig, n_steps: int):
    return loss_registry.build(
        config.name,
        inject_dict(config.config, {"n_steps": n_steps})
    )


def _mitsuba_post_init(cfg: Union[dict, DictConfig], obj: Any, scene: mi.Scene, injection: dict[str, Any] = None, kwargs_injection: dict[str, Any] = None):
    # NOTE: for unknown reasons, torch module creation must
    # happen after mitsuba object contruction (not during).
    # Therefore, we have this post_init hack.

    if not hasattr(obj, "post_init"):
        return

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg)
    assert isinstance(cfg, dict)

    if "kwargs" in cfg:
        bbox = scene.bbox()

        kwargs_injection = kwargs_injection or {}
        kwargs_injection.update({
            "scene_min": bbox.max,
            "scene_max": bbox.min,
        })
        inject_dict(cfg["kwargs"], kwargs_injection)

    injection = injection or {}
    inject_dict(cfg, injection)

    obj.post_init(**cfg)


def prepare_learned_objects(
    scene: mi.Scene,
    integrator: mi.Integrator,
    learned_modules: dict[str, nn.Module],
    envmap_cfg: EnvmapConfig,
    train_cfg: Optional[TrainConfig],
    ckpt_file: Optional[str],
    radiance_cache_ckpt: Optional[str],
    device: str,
) -> dict[str, Any]:
    result = {}

    # Add params to mi for gradient flow
    params = mi.traverse(scene)

    # NOTE: we should not add new keys to params, instead, we should have created the integrator
    # within the scene hierarchy
    # However, this is what it is so we're doing this hack for now
    integrator_params = mi.traverse(integrator)

    # Register learned integrator
    if isinstance(integrator, nn.Module):
        learned_modules["integrator"] = integrator

    # Remove learned modules without parameter
    learned_modules = {
        k: v for k, v in learned_modules.items() if len(list(v.parameters())) > 0
    }

    logger.info(
        "Learned modules:\n" + ",\n".join([f"{name}: {obj}" for name, obj in learned_modules.items()])
    )

    # Create Mitsuba optimizer

    # Hard-coded: all possible key suffixes that requires dr.jit and mitsuba handling
    dr_grad_keys = [
        ".grad_activator",
        ".tensor",
    ]
    mi_optimized_keys = [
        ".mi_texture",
    ]

    # Hack: to learn envmap, using mitsuba bitmap for now
    assert envmap_cfg.name in {"gt", "mitsuba"}
    learn_envmap = envmap_cfg.name == "mitsuba"
    result["learn_envmap"] = learn_envmap
    if learn_envmap:
        logger.info("Enable envmap training")
        envmap_key = find_envmap_param_key(params, scene)
        mi_optimized_keys.append(envmap_key)

        gt_envmap: mi.TensorXf = params[envmap_key]

        envmap_shape = [
            envmap_cfg.config.get("height") or gt_envmap.shape[0],
            envmap_cfg.config.get("width") or gt_envmap.shape[1],
            gt_envmap.shape[2],
        ]
        logger.info(f"Learned envmap shape: {envmap_shape}")
        params[envmap_key] = dr.full(mi.TensorXf, 1, envmap_shape)
        params.update()

        result.update({
            "gt_envmap": gt_envmap,
            "envmap_key": envmap_key,
        })

    mi_optimized_params = {}
    for key, obj in params.items():
        if any((key.endswith(s) for s in dr_grad_keys)):
            logger.info(f"dr.enable_grad: {key}")
            dr.enable_grad(obj)
            continue

        if any((key.endswith(s) for s in mi_optimized_keys)):
            logger.info(f"Trained with Mitsuba: {key}")
            dr.enable_grad(obj)
            mi_optimized_params[key] = obj
            continue

    for key, obj in integrator_params.items():
        if key == "grad_activator":
            logger.info(f"dr.enable_grad: integrator {key}")
            dr.enable_grad(obj)
            continue

        if key == "mi_texture":
            logger.info(f"Trained with Mitsuba: integrator {key}")
            dr.enable_grad(obj)
            mi_optimized_params[key] = obj
            continue

    mi_optim = None
    if train_cfg is not None and len(mi_optimized_params) > 0:
        mi_optim = mi.ad.Adam(lr=train_cfg.learning_rate, beta_1=train_cfg.beta_1, beta_2=train_cfg.beta_2)
        for key, obj in mi_optimized_params.items():
            mi_optim[key] = obj
        params.update(mi_optim)
        integrator_params.update(mi_optim)

    # Create PyTorch optimizer

    torch_optimized_params = []
    for key, obj in learned_modules.items():
        logger.info(f"Trained with PyTorch: {key}")
        obj.to(device)
        torch_optimized_params += list(obj.parameters())

    torch_optim = None
    if train_cfg is not None and len(torch_optimized_params) > 0:
        torch_optim = torch.optim.Adam(torch_optimized_params, lr=train_cfg.learning_rate,
                                       betas=(train_cfg.beta_1, train_cfg.beta_2))

    logger.info(
        "Optimizer summary:\n"
        f"Mitsuba: {mi_optim is not None} ({len(mi_optimized_params)})\n"
        f"PyTorch: {torch_optim is not None} ({len(torch_optimized_params)})"
    )

    # LR scheduling
    mi_scheduler = None
    torch_scheduler = None
    if train_cfg is not None and train_cfg.lr_decay_start >= 0:
        lr_scheduler_args = (train_cfg.lr_decay_start, train_cfg.lr_decay_rate,
                             train_cfg.lr_decay_steps, train_cfg.lr_decay_min_rate)
        if mi_optim is not None:
            mi_scheduler = ExpLR(*lr_scheduler_args)
        if torch_optim is not None:
            torch_scheduler = ExpLRScheduler(torch_optim, *lr_scheduler_args)

    # Resume training
    result["step"] = 0
    if ckpt_file is not None:
        logger.info(f"Load checkpoint {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        last_step = ckpt["step"]
        result["step"] = last_step
        logger.info(f"Checkpoint step is {last_step}")

        if torch_optim is not None:
            logger.info("Load torch optim")
            torch_optim.load_state_dict(ckpt["optim"])

        for name, obj in learned_modules.items():
            logger.info(f"Load torch module {name}")
            obj.load_state_dict(ckpt["modules"][name])

        if torch_scheduler is not None:
            if "scheduler" in ckpt:
                torch_scheduler.load_state_dict(ckpt["scheduler"])
            else:
                torch_scheduler.last_epoch = last_step - 1

        if mi_optim is not None:
            logger.info("Load mi optim")
            mi_optim.state.update({
                k: tuple(mi.TensorXf(v.to(device)) for v in s) for k, s in ckpt["mi_optim"].items()
            })

        for name in mi_optimized_params:
            logger.info(f"Load mi param {name}")
            data = mi.TensorXf(ckpt["mi_params"][name].to(device))
            if name in params:
                params[name] = data
            elif name in integrator_params:
                integrator_params[name] = data
            if mi_optim is not None:
                mi_optim[name] = data
            mi_optimized_params[name] = data

        if len(mi_optimized_params) > 0:
            params.update(mi_optim)
            integrator_params.update(mi_optim)

        if mi_scheduler is not None:
            mi_optim.set_learning_rate(mi_scheduler.get_lr_rate(last_step - 1) * train_cfg.learning_rate)

    if radiance_cache_ckpt is not None:
        #load a pre-trained radiance cache, that is most likely trained using GT truth data directly, similar to Zhang et. al
        radiance_ckpt = torch.load(radiance_cache_ckpt, map_location="cpu")
        name = "integrator"
        logger.info(f"Load radiance cache for torch module {name}")
        learned_modules[name].load_state_dict(radiance_ckpt["modules"][name])
    result.update({
        "params": params,
        "integrator_params": integrator_params,
        "learned_modules": learned_modules,
        "mi_optimized_params": mi_optimized_params,
        "mi_optim": mi_optim,
        "torch_optim": torch_optim,
        "mi_scheduler": mi_scheduler,
        "torch_scheduler": torch_scheduler,
    })

    return result


def find_envmap_param_key(params: mi.SceneParameters, scene: mi.Scene) -> str:
    env_emitters = [em for em in scene.emitters() if em.is_environment()]
    if len(env_emitters) != 1:
        raise ValueError(f"Expecting 1 environment map in the scene, found {len(env_emitters)}")

    data = mi.traverse(env_emitters[0])["data"]
    for key, obj in params:
        if data is obj:
            return key

    raise ValueError("Couldn't find environment map data in scene params")


def compute_output_metrics(
    name: str,
    outputs: list[mi.Bitmap],
    integrator: str,
    gt: dict[str, mi.Bitmap],
):
    gt = gt.get(name)
    if gt is None:
        return {}

    is_image = name == "image"
    names = [name]
    if is_image and integrator.startswith("nerad"):
        names = ["lhs", "rhs"]
    assert len(names) == len(outputs)

    results = {}
    for name, pred in zip(names, outputs):
        metrics = compute_metrics(pred, gt)
        for key, value in metrics.items():
            results[f"{name}_{key}"] = value

    return results
