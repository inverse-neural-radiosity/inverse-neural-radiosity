DATA=data
OUT=output/ablation/lego

bash ./scripts/train-nerf-dataset/run_principled.sh \
    out_root=$OUT/ \
    dataset.scene=$DATA/nerf_scenes/lego/scene_principled.xml \
    dataset.cameras=$DATA/datasets/lego/transforms.json \
    dataset.albedo=$DATA/datasets/lego/albedo/exr \
    dataset.roughness=$DATA/datasets/lego/roughness/exr \
    validation_view=9 \
    ${@}
