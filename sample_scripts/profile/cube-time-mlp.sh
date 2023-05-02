DATA=/data/nerad

MLP_NERAD="rendering=nerad_sg"
NERAD_1B="rendering.integrator=nerad_1b"
MYPATH="rendering=mypath_unlimited"
MYPRB="rendering=myprb_unlimited"

MLP_BSDF="bsdf=diffuse_fake_mlp"

MEGAKERNEL="compute.dr_optimization_flags=true"

declare -A LIMITS
LIMITS["0.3"]=5
LIMITS["0.5"]=7
LIMITS["0.7"]=13
LIMITS["0.9"]=42
LIMITS["0.95"]=84
LIMITS["0.97"]=140


function run {
python watchdog.py --max_retries 1 --timeout 1800 -- \
    --config-name profile_train \
    out_root=/output/nerad/cube_time_mlp/$ITER/cube_scene-path_limit_0/albedo_${ALBEDO} \
    dataset.scene=/data/nerad/mitsuba3_scenes/cube_scene/scene_diffuse.xml \
    dataset.cameras=/data/nerad/datasets/cube_scene/albedo_${ALBEDO}/transforms.json \
    bsdf.texture.reflectance.post_init.kwargs.value=${ALBEDO} \
    ${@} \
    batch_size=128 \
    rendering.spp=16 \
    disable_optimizer_step=true
}

for ITER in 01 02 03 04 05; do

### Ours 1B

LIMIT=1
for ALBEDO in 0.3 0.5 0.7 0.9 0.95 0.97; do
    run \
        $MLP_BSDF \
        $MLP_NERAD \
        $NERAD_1B \
        name=nerad1b
done

### Ours

LIMIT=2
for ALBEDO in 0.3 0.5 0.7 0.9 0.95 0.97; do
    run \
        $MLP_BSDF \
        $MLP_NERAD \
        name=nerad
done

### PRB

for ALBEDO in 0.3 0.5 0.7 0.9 0.95 0.97; do
    LIMIT=${LIMITS[$ALBEDO]}
    run \
        $MLP_BSDF \
        $MYPRB \
        name=prb \
        rendering.config.max_depth=$LIMIT \
        rendering.config.rr_prob=0.95
done

### Path

for ALBEDO in 0.3 0.5 0.7 0.9 0.95 0.97; do
    LIMIT=${LIMITS[$ALBEDO]}
    run \
        $MLP_BSDF \
        $MYPATH \
        name=path \
        rendering.config.max_depth=$LIMIT \
        rendering.config.rr_prob=0.95
done

done
