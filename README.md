Inverse Global Illumination using a Neural Radiometric Prior
---

Code release for the paper, *Inverse Global Illumination using a Neural Radiometric Prior*, accepted to SIGGRAPH 2023 Conference Proceedings.

[Project Homepage](https://inverse-neural-radiosity.github.io/)

```
Saeed Hadadan, University of Maryland, College Park & NVIDIA
Geng Lin, University of Maryland, College Park
Jan Novák, NVIDIA
Fabrice Rousselle, NVIDIA
Matthias Zwicker, University of Maryland, College Park
```

## Environment Setup

Prepare an environment with CUDA 11.7.
Then, in virtualenv or Conda, install PyTorch and other dependencies:

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r init/requirements.txt
```

Newer versions of CUDA and PyTorch should work but have not been tested.

### Installing OpenEXR

1. Linux users should install `libopenexr-dev`
1. Windows users should use Conda and run `conda install -c conda-forge openexr`

## How to Train

Download our data from [Google Drive](https://drive.google.com/drive/folders/1V-LVNWWhIVmhF3n2xflYFuwLiQRvVcW0?usp=sharing).

Training scripts can be found in `./sample_scripts`. Copy them to `./scripts` and edit the data and output paths.

As an exmple, to reconstruct the Staircase scene, run **in this folder**:

```bash
source ./init/init.source  # do this once per shell

bash ./scripts/train-staircase/nerad_principled.sh  # our method
bash ./scripts/train-staircase/prb_principled.sh  # PRB
```

## How to Evaluate

Suppose the training folder is `/output/nerad/staircase/2023-04-25-06-21-10-nerad`, simply run:

```bash
python test.py \
    test_rendering.image.spp=512 \
    test_rendering.albedo.spp=512 \
    test_rendering.roughness.spp=512 \
    experiment=/output/nerad/staircase/2023-04-25-06-21-10-nerad
```

All views in the dataset is rendered to `$TRAINING_FOLDER/test/latest`. Check `test.py` and `nerad/model/config.py` for available options.

## Cite

```bibtex
@misc{hadadan2023inverse,
      title={Inverse Global Illumination using a Neural Radiometric Prior},
      author={Saeed Hadadan and Geng Lin and Jan Novák and Fabrice Rousselle and Matthias Zwicker},
      year={2023},
      eprint={2305.02192},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
