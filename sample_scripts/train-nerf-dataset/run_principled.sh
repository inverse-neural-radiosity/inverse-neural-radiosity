python watchdog.py --max_retries 1200 -- \
    bsdf=principled \
    'saving=[latest]' \
    'validation=[image, roughness, albedo]' \
    batch_size=64 \
    learning_rate=0.0005 \
    rendering.spp=64 \
    validation.albedo.step_size=500 \
    validation.image.step_size=500 \
    validation.roughness.step_size=500 \
    n_steps=10000 \
    lr_decay_start=5000 \
    lr_decay_rate=0.5 \
    lr_decay_steps=2500 \
    lr_decay_min_rate=0.1 \
    avoid_empty_batches=true \
    ${@}
