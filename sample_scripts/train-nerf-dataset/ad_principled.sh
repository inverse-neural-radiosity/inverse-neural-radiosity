SCENE=$1
bash ./scripts/train-nerf-dataset/run_principled_$SCENE.sh \
rendering=mypath \
rendering.config.max_depth=15 \
name=ad
