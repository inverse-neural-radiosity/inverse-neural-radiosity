DATA=data
OUT=output/nerad_revised_nerf/hotdog

bash ./scripts/train-nerf-dataset/run_principled.sh \
    out_root=$OUT/ \
    dataset.scene=$DATA/nerf_scenes/hotdog/scene_principled.xml \
    dataset.cameras=$DATA/datasets/hotdog/transforms.json \
    dataset.albedo=$DATA/datasets/hotdog/albedo/exr \
    dataset.roughness=$DATA/datasets/hotdog/roughness/exr \
    validation_view=24 \
    ${@}
