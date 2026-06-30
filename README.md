# GFSS-EKT

Official PyTorch implementation of **Enhancing Generalized Few-Shot Semantic
Segmentation via Effective Knowledge Transfer** (AAAI 2025).

[[Paper](https://arxiv.org/abs/2412.15835)]

GFSS-EKT performs generalized few-shot semantic segmentation (GFSS): after
learning from fully annotated base classes, the model is adapted with only a few
annotated examples from novel classes and must segment both base and novel
classes at test time.

The training pipeline has two stages:

1. **Base training** (`train_base.py`): train the feature extractor, base
   prototypes, and base classifier on the base classes.
2. **Few-shot fine-tuning** (`ft_ekt.py`): load the base checkpoint and learn
   the novel prototypes and classifier using GFSS-EKT's NPM, NCC, and CCL
   components.

The three components are Novel Prototype Modulation (NPM), Novel Classifier
Calibration (NCC), and Context Consistency Learning (CCL).

## Repository layout

```text
GFSS-EKT/
├── dataset/                 # dataset classes, CLIP tokenizer, and split files
│   └── list/
│       ├── voc/
│       └── coco/
├── loss/                    # segmentation and orthogonality losses
├── networks/                # PSPNet, backbones, CLIP modules, GFSS-EKT modules
├── scripts/                 # example SLURM/shell launchers
├── utils/                   # metrics and data-preparation utilities
├── engine.py                 # distributed-training engine used by train_base.py/ft_ekt.py
├── train_base.py             # stage 1: base-class training
├── ft_ekt.py                 # stage 2: few-shot fine-tuning and validation
├── eval_ft.py                 # standalone inference/visualization script
├── tsne.py                    # t-SNE/UMAP feature embedding visualization
└── visualization_weight_create.py  # classifier weight-distribution plots
```

## 1. Environment

The code requires Linux, an NVIDIA GPU, and CUDA. The launch scripts were
written for Python 3.8 and use legacy PyTorch/MMCV APIs. The following is a
reference environment; select PyTorch and MMCV builds that match your CUDA
installation.

```bash
conda create -n gfss-ekt python=3.8 -y
conda activate gfss-ekt

# Example for CUDA 11.1
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

pip install mmsegmentation==0.30.0 \
  numpy==1.23.5 scipy==1.10.1 opencv-python==4.7.0.72 \
  Pillow==9.5.0 matplotlib==3.7.5 pycocotools \
  ftfy==6.1.1 regex timm==0.6.13 transformers==4.26.1
```

Clone the repository:

```bash
git clone https://github.com/HHHHedy/GFSS-EKT.git
cd GFSS-EKT
```

## 2. Dataset preparation

### PASCAL-5i

Download:

- [PASCAL VOC 2012 train/validation data](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/)
- [Semantic Boundaries Dataset (SBD)](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)

Extract VOC 2012 and SBD:

```bash
mkdir -p data
cd data

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar

wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
tar -xf benchmark.tgz
cd ..
```

Create the augmented semantic masks used by `trainaug.txt`. This converts the
SBD MATLAB masks to indexed PNG files and then copies the official VOC masks on
top:

```bash
export VOC_ROOT="$PWD/data/VOCdevkit/VOC2012"
export SBD_ROOT="$PWD/data/benchmark_RELEASE/dataset"

python - <<'PY'
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

voc_root = Path(os.environ["VOC_ROOT"])
sbd_root = Path(os.environ["SBD_ROOT"])
output = voc_root / "SegmentationClassAug"
output.mkdir(parents=True, exist_ok=True)

for mat_path in (sbd_root / "cls").glob("*.mat"):
    data = loadmat(mat_path)
    mask = data["GTcls"][0]["Segmentation"][0].astype(np.uint8)
    Image.fromarray(mask).save(output / f"{mat_path.stem}.png")

for mask_path in (voc_root / "SegmentationClass").glob("*.png"):
    shutil.copy2(mask_path, output / mask_path.name)
PY
```

The resulting layout must be:

```text
data/VOCdevkit/VOC2012/
├── JPEGImages/
│   ├── 2007_000032.jpg
│   └── ...
├── SegmentationClass/
├── SegmentationClassAug/
│   ├── 2007_000032.png
│   └── ...
└── ...
```

Check that every repository split entry has an image and a mask:

```bash
python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["VOC_ROOT"])
for split in ("dataset/list/voc/trainaug.txt", "dataset/list/voc/val.txt"):
    ids = Path(split).read_text().splitlines()
    missing_images = [x for x in ids if not (root / "JPEGImages" / f"{x}.jpg").is_file()]
    missing_masks = [x for x in ids if not (root / "SegmentationClassAug" / f"{x}.png").is_file()]
    print(split, f"images={len(ids)}, missing_images={len(missing_images)}, missing_masks={len(missing_masks)}")
    assert not missing_images and not missing_masks
PY
```

The provided VOC lists contain 10,582 training images and 1,449 validation
images. Fold files for seed 123 and common shot settings are already included
under `dataset/list/voc/fold*/`.

### COCO-20i

Download the following files from the
[COCO download page](https://cocodataset.org/#download):

- 2014 train images
- 2014 validation images
- 2014 train/validation annotations

They can also be downloaded directly:

```bash
mkdir -p data/COCO2014
cd data/COCO2014

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip train2014.zip
unzip val2014.zip
unzip annotations_trainval2014.zip
cd ../..
```

Extract them into a common root:

```text
data/COCO2014/
├── train2014/
├── val2014/
└── annotations/
    ├── instances_train2014.json
    └── instances_val2014.json
```

GFSS-EKT reads class-indexed PNG masks rather than COCO JSON annotations.
Generate them with `utils/coco_parse_script.py`:

1. Set `root_dir` to the absolute `data/COCO2014` path.
2. Set `partition = 'train'` and run the script.
3. Set `partition = 'val'` and run it again.
4. Move the generated directories to the paths expected by the data loader.

```bash
python utils/coco_parse_script.py  # partition='train'
python utils/coco_parse_script.py  # partition='val'

mv data/COCO2014/train_no_crowd data/COCO2014/annotations/train2014
mv data/COCO2014/val_no_crowd data/COCO2014/annotations/val2014
```

The final layout must be:

```text
data/COCO2014/
├── train2014/
│   ├── COCO_train2014_000000000009.jpg
│   └── ...
├── val2014/
├── annotations/
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   ├── train2014/
│   │   ├── COCO_train2014_000000000009.png
│   │   └── ...
│   └── val2014/
└── ...
```

Generating the COCO masks and per-class fold lists requires scanning the full
dataset and can take some time.

Create the per-class metadata consumed by `utils/gen_fs_list.py`:

```bash
export COCO_ROOT="$PWD/data/COCO2014"

python - <<'PY'
import os

from dataset.coco import GFSSegTrain

for fold in range(4):
    GFSSegTrain(
        root=os.environ["COCO_ROOT"],
        list_path="dataset/list/coco/train.txt",
        fold=fold,
        mode="train",
    )
PY
```

This creates `dataset/list/coco/fold0/` through `fold3/`. Generate the
seed-specific few-shot files after this step as described in
[Stage 2](#stage-2-few-shot-fine-tuning).

## 3. Training

All commands below must be run from the repository root. Both `train_base.py`
and `ft_ekt.py` take their configuration via `argparse` command-line flags —
there are no environment variables to set, and running them with no flags
falls back to unrelated defaults (`--dataset cityscapes`, `--data-dir
/data/pascal-context`). Use the flags below, or edit and run the matching
script under `scripts/` (those also contain SLURM headers and absolute paths
meant for a cluster, so adjust them for local use).

### Stage 1: train on base classes

PASCAL-5i, fold 0 (see [scripts/train_voc_fold0_base.sh](scripts/train_voc_fold0_base.sh)):

```bash
python train_base.py --dataset voc \
  --data-dir /absolute/path/to/data/VOCdevkit/VOC2012 \
  --train-list ./dataset/list/voc/trainaug.txt \
  --val-list ./dataset/list/voc/val.txt \
  --random-seed 123 \
  --model pspnet_ekt --backbone resnet50v2 \
  --restore-from /absolute/path/to/backbones/resnet50_v2.pth \
  --input-size 473,473 --base-size 473,473 \
  --learning-rate 1e-3 --weight-decay 1e-4 \
  --batch-size 8 --test-batch-size 8 \
  --start-epoch 0 --num-epoch 50 --os 8 \
  --snapshot-dir ./exp/voc/pspnet_ekt/0/resnet50v2/base/baseline \
  --save-pred-every 50 --fold 0
```

The best base checkpoint is saved to `<snapshot-dir>/best.pth`.

Repeat with `--fold 1`, `2`, and `3` for four-fold evaluation.

### Stage 2: few-shot fine-tuning

PASCAL-5i, fold 0, 1-shot (see [scripts/ft_voc.sh](scripts/ft_voc.sh)):

```bash
python ft_ekt.py --dataset voc \
  --data-dir /absolute/path/to/data/VOCdevkit/VOC2012 \
  --train-list ./dataset/list/voc/trainaug.txt \
  --val-list ./dataset/list/voc/val.txt \
  --random-seed 123 \
  --model pspnet_ekt --backbone resnet50v2 \
  --restore-from ./exp/voc/pspnet_ekt/0/resnet50v2/base/baseline/best.pth \
  --input-size 473,473 --base-size 473,473 \
  --learning-rate 1e-3 --weight-decay 1e-4 \
  --batch-size 1 --test-batch-size 16 \
  --start-epoch 0 --num-epoch 500 --os 8 \
  --snapshot-dir ./exp/voc/pspnet_ekt/0/resnet50v2/novel/1shot \
  --save-pred-every 50 --fold 0 --shot 1 \
  --freeze-backbone --fix-lr --update-base --update-epoch 1
```

`--restore-from` must point at the stage-1 checkpoint for the same fold and
backbone.

## 4. Testing

`ft_ekt.py` runs validation during fine-tuning, reports the current and best
scores, and selects `best_<seed>.pth` using overall mIoU.

For each class \(c\):

```text
IoU(c) = intersection(c) / union(c)
```

The evaluation metrics are:

- **Base mIoU**: mean IoU over the background and base classes.
- **Novel mIoU**: mean IoU over the novel classes.
- **Mean mIoU**: mean IoU over the background, base, and novel classes.
- **H-Mean**: `2 * Base * Novel / (Base + Novel)`.

The training code prints Base, Novel, and Mean. Compute H-Mean from the
four-fold averaged Base and Novel scores to reproduce the paper protocol.

Run both training stages for folds 0–3 and average the fold results. PASCAL-5i
uses 15 base and 5 novel classes per fold; COCO-20i uses 60 base and 20 novel
classes per fold.

## Acknowledgements

This codebase is built upon [POP](https://github.com/lsa1997/pop)
(*Learning Orthogonal Prototypes for Generalized Few-shot Semantic
Segmentation*, CVPR 2023). If you use this code, please also cite POP:

```bibtex
@inproceedings{liu2023learning,
  title     = {Learning Orthogonal Prototypes for Generalized Few-shot Semantic Segmentation},
  author    = {Liu, Sun-Ao and Zhang, Yiheng and Qiu, Zhaofan and Xie, Hongtao and Zhang, Yongdong and Yao, Ting},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2023}
}
```

## Citation

If this project is useful in your research, please cite:

```bibtex
@inproceedings{chen2025enhancing,
  title={Enhancing generalized few-shot semantic segmentation via effective knowledge transfer},
  author={Chen, Xinyue and Shi, Miaojing and Zhou, Zijian and He, Lianghua and Tsoka, Sophia},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={2},
  pages={2256--2265},
  year={2025}
}
```

## License

See [LICENSE](LICENSE).

