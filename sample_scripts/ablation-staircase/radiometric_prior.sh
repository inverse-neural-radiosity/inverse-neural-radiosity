bash ./scripts/ablation/run_principled.sh \
rendering=nerad \
name=radiometric_prior \
rendering.config.second_residual=false \
rendering.config.detach_non_radiance_gradients_in_residual=false \
LHS_recons_loss=none \
