SCENE=$1
bash ./scripts/train-nerf-dataset/run_principled_$SCENE.sh \
rendering=nerad \
name=zhang \
residual_loss=none \
