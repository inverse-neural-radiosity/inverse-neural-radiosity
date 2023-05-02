DATA=data
OUT=output/nerad_revised_nerf/ficus

bash ./scripts/train-nerf-dataset/run_principled.sh \
    out_root=$OUT/ \
    dataset.scene=$DATA/nerf_scenes/ficus/scene_principled.xml \
    dataset.cameras=$DATA/datasets/ficus/transforms.json \
    dataset.albedo=$DATA/datasets/ficus/albedo/exr \
    dataset.roughness=$DATA/datasets/ficus/roughness/exr \
    validation_view=9 \
    ${@}
