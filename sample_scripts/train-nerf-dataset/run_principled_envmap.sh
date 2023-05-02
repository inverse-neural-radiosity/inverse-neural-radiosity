DATA=data
OUT=output/nerad/mat_lighting

python watchdog.py -- \
    out_root=$OUT/ \
    dataset.scene=$DATA/nerf_scenes/lego/scene_principled.xml \
    dataset.cameras=$DATA/datasets/lego/transforms.json \
    bsdf=principled \
    'saving=[checkpoint, latest]' \
    saving.checkpoint.step_size=5000 \
    'validation=[image, roughness, albedo]' \
    batch_size=64 \
    learning_rate=0.0005 \
    rendering.spp=64 \
    validation.albedo.step_size=2000 \
    validation.image.step_size=2000 \
    validation.roughness.step_size=2000 \
    n_steps=10000 \
    envmap=mitsuba \
    envmap.config.width=256 \
    envmap.config.height=128 \
    validation_view=12 \
    ${@}
