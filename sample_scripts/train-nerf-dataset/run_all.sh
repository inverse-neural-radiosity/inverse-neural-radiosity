SCENE=lego
bash ./scripts/train-nerf-dataset/nerad_principled.sh $SCENE
bash ./scripts/train-nerf-dataset/prb_principled.sh $SCENE
bash ./scripts/train-nerf-dataset/zhang_principled.sh $SCENE
bash ./scripts/train-nerf-dataset/ad_principled.sh $SCENE
bash ./scripts/train-nerf-dataset/direct_principled.sh $SCENE


SCENE=hotdog
bash ./scripts/train-nerf-dataset/nerad_principled.sh $SCENE
bash ./scripts/train-nerf-dataset/prb_principled.sh $SCENE
bash ./scripts/train-nerf-dataset/zhang_principled.sh $SCENE
bash ./scripts/train-nerf-dataset/ad_principled.sh $SCENE
bash ./scripts/train-nerf-dataset/direct_principled.sh $SCENE

SCENE=ficus
bash ./scripts/train-nerf-dataset/nerad_principled.sh $SCENE
bash ./scripts/train-nerf-dataset/prb_principled.sh $SCENE
bash ./scripts/train-nerf-dataset/zhang_principled.sh $SCENE
bash ./scripts/train-nerf-dataset/ad_principled.sh $SCENE
bash ./scripts/train-nerf-dataset/direct_principled.sh $SCENE
