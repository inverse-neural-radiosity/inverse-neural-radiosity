SCENE=$1
bash ./scripts/train-nerf-dataset/run_principled_$SCENE.sh \
rendering=myprb \
rendering.config.max_depth=15 \
rendering.config.allow_eval_direction_call=false \
name=prb \
