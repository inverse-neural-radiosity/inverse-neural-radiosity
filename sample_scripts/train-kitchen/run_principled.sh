DATA=data
OUT=output/nerad_revised/kitchenexps

python watchdog.py -- \
    out_root=$OUT/ \
    dataset.scene=$DATA/mitsuba3_scenes/kitchen/scene_principled_one_area.xml \
    dataset.cameras=$DATA/datasets/kitchen_one_area/transforms.json \
    bsdf=principled \
    'saving=[checkpoint, latest]' \
    saving.checkpoint.step_size=16000 \
    'validation=[albedo, roughness, image]' \
    batch_size=64 \
    learning_rate=0.0005 \
    lr_decay_start=100000   \
    rendering.spp=64 \
    validation.roughness.step_size=8000 \
    validation.image.step_size=8000 \
    validation.albedo.step_size=8000 \
    n_steps=32000 \
    dataset.albedo=$DATA/datasets/kitchen_one_area/albedo/exr \
    dataset.roughness=$DATA/datasets/kitchen_one_area/roughness/exr \
    ${@}
