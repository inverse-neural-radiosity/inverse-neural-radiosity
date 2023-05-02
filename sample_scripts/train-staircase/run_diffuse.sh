DATA=data
OUT=output/nerad/staircase

python watchdog.py --max_retries 120 -- \
    out_root=$OUT \
    dataset.scene=$DATA/mitsuba3_scenes/staircase/scene_diffuse.xml \
    dataset.cameras=$DATA/datasets/staircase/transforms.json \
    bsdf=diffuse \
    'saving=[checkpoint, latest]' \
    saving.checkpoint.step_size=1500 \
    'validation=[albedo, image]' \
    batch_size=128 \
    dataset.n_views=26 \
    learning_rate=0.0005 \
    rendering.spp=16 \
    validation.image.step_size=1000 \
    validation.albedo.step_size=1000 \
    n_steps=6000 \
    dataset.albedo=$DATA/datasets/staircase/albedo/exr \
    ${@}
