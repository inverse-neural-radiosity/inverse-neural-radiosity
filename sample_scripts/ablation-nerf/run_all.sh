SCENE=lego
bash ./scripts/ablation-nerf/radiometric_prior.sh $SCENE
bash ./scripts/ablation-nerf/second_residual.sh $SCENE
bash ./scripts/ablation-nerf/stop_gradient.sh $SCENE
bash ./scripts/ablation-nerf/complete.sh $SCENE
bash ./scripts/ablation-nerf/zhang.sh $SCENE


SCENE=hotdog
bash ./scripts/ablation-nerf/radiometric_prior.sh $SCENE
bash ./scripts/ablation-nerf/second_residual.sh $SCENE
bash ./scripts/ablation-nerf/stop_gradient.sh $SCENE
bash ./scripts/ablation-nerf/complete.sh $SCENE
bash ./scripts/ablation-nerf/zhang.sh $SCENE
