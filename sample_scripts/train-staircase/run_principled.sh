DATA=data
OUT=output/nerad_revised/staircase

python watchdog.py --max_retries 120 -- \
    out_root=$OUT \
    dataset.scene=$DATA/mitsuba3_scenes/staircase/scene_principled.xml \
    dataset.cameras=$DATA/datasets/staircase_principled/transforms.json \
    bsdf=principled \
    'saving=[latest]' \
    'validation=[albedo, roughness, image]' \
    batch_size=128 \
    dataset.n_views=26 \
    learning_rate=0.0005 \
    rendering.spp=16 \
    validation.roughness.step_size=6000 \
    validation.image.step_size=6000 \
    validation.albedo.step_size=6000 \
    n_steps=18000 \
    dataset.albedo=$DATA/datasets/staircase_principled/albedo/exr \
    dataset.roughness=$DATA/datasets/staircase_principled/roughness/exr \
    ${@}
