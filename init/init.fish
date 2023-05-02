# disable core dump
ulimit -c 0

# make Python work better
set -x PYTHONPATH (pwd)

# use GPU
set -x MI_DEFAULT_VARIANT cuda_ad_rgb
set -x TORCH_DEVICE cuda:0
