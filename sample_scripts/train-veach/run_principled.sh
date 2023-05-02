DATA=data
OUT=output/nerad_revised/veach_ajar
python watchdog.py -- \
    out_root=$OUT/ \
    dataset.scene=$DATA/mitsuba3_scenes/veach_ajar/scene_principled.xml \
    dataset.cameras=$DATA/datasets/veach_ajar/transforms.json \
    bsdf=principled \
    'saving=[latest]' \
    saving.latest.step_size=100 \
    'validation=[albedo, roughness, image]' \
    batch_size=64 \
    learning_rate=0.0005 \
    lr_decay_start=100000   \
    rendering.spp=64 \
    validation.roughness.step_size=8000 \
    validation.image.step_size=8000 \
    validation.albedo.step_size=8000 \
    n_steps=32000 \
    dataset.albedo=$DATA/datasets/veach_ajar/albedo/exr \
    dataset.roughness=$DATA/datasets/veach_ajar/roughness/exr \
    ${@}
