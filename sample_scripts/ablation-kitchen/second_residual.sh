bash ./scripts/ablation-kitchen/run_principled.sh \
rendering=nerad \
name=second_residual \
rendering.config.second_residual=true \
rendering.config.detach_non_radiance_gradients_in_residual=true \
LHS_recons_loss=none \
