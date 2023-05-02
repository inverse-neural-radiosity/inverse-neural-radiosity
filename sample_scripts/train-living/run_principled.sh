DATA=data
OUT=output/nerad_revised/living-room-2

python watchdog.py -- \
    out_root=$OUT/ \
    dataset.scene=$DATA/mitsuba3_scenes/living-room-2/scene_principled.xml \
    dataset.cameras=$DATA/datasets/living-room-2/transforms.json \
    bsdf=principled \
    'saving=[latest]' \
    saving.latest.step_size=100 \
    'validation=[albedo, roughness, image]' \
    batch_size=64 \
    learning_rate=0.0005 \
    lr_decay_start=100000   \
    rendering.spp=64 \
    validation.roughness.step_size=2000 \
    validation.image.step_size=2000 \
    validation.albedo.step_size=2000 \
    n_steps=32000 \
    dataset.albedo=$DATA/datasets/living-room-2/albedo/exr \
    dataset.roughness=$DATA/datasets/living-room-2/roughness/exr \
    ${@}
