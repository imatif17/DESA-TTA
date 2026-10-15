# DESA-TTA: Dynamic EMA and Source Anchoring for Test-Time Adaptation

**Atif Belal, Lilian Hollard, Marco Pedersoli, Eric Granger**

LIVIA, ILLS, Dept. of Systems Engineering, ÉTS Montréal, Canada

[![arXiv](https://img.shields.io/badge/arXiv-2609.01795-b31b1b.svg)](https://arxiv.org/abs/2609.01795)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Official PyTorch implementation of DESA-TTA.

## Overview

Vision-language object detectors (VLODs) such as YOLO-World and Grounding DINO detect well zero-shot, but lose accuracy under distribution shift. Mean-teacher test-time adaptation (TTA) helps, but it relies on a fixed EMA coefficient for the teacher and drifts over time because of noisy pseudo-labels. DESA-TTA adds two components to the mean teacher:

- **Dynamic EMA (D-EMA).** The teacher momentum is chosen per test image from the teacher's uncertainty, estimated from pseudo-label confidence and box density. The range it can take is set by how far the teacher has drifted from the source model.
- **Source anchoring.** After each update, the student is pulled back towards its pretrained parameters, with a strength that grows with the student's drift.

Only normalization parameters are adapted (BatchNorm for YOLO-World), which is about 0.02M parameters for YOLO-World-S.

## Results

**VOC-C, YOLO-World-S** (AP50 under 15 corruptions):

| Method | Gau. | Shot | Imp. | Def. | Glass | Mot. | Zoom | Snow | Frost | Fog | Bright. | Cont. | Elast. | Pix. | JPEG | **Avg.** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Zero-shot | 16.9 | 17.2 | 16.2 | 32.4 | 11.3 | 26.9 | 30.1 | 46.4 | 50.6 | 70.4 | 73.6 | 41.3 | 42.0 | 8.9 | 26.1 | 34.0 |
| Mean teacher | 27.1 | 28.3 | 28.0 | 33.7 | 14.8 | 28.3 | 30.3 | 51.5 | 54.2 | 69.0 | 70.0 | 55.8 | 51.5 | 10.9 | 39.4 | 39.5 |
| VLOD-TTA | 22.9 | 24.4 | 24.2 | 37.2 | 16.6 | 29.3 | 30.9 | 50.9 | 53.8 | 73.2 | 74.1 | 48.7 | 49.4 | 17.8 | 40.2 | 39.6 |
| **DESA-TTA** | **34.4** | **35.8** | **37.8** | **39.6** | **32.7** | **37.8** | **36.1** | **54.5** | **57.8** | **73.6** | **74.2** | **63.3** | **59.8** | **37.9** | **52.6** | **48.5** |

On VOC-C, DESA-TTA runs at 31 FPS on an RTX 3090, 55% faster than VLOD-TTA (20 FPS).

**Other benchmarks** (code coming soon):

| Benchmark | Detector | Metric | Zero-shot | DESA-TTA |
|---|---|---|---|---|
| Watercolor | YOLO-World-S | AP50 | 47.9 | 55.0 |
| ClipArt | YOLO-World-S | AP50 | 40.1 | 48.3 |
| Comic | YOLO-World-S | AP50 | 29.4 | 38.6 |
| Foggy Cityscapes | YOLO-World-S | AP50 | 21.3 | 24.4 |
| BDD100K | YOLO-World-S | AP50 | 22.0 | 27.8 |
| ExDark | YOLO-World-S | AP50 | 64.7 | 69.2 |
| COCO-C | Grounding DINO | mAP | 20.6 | 27.6 |

## Installation

The code builds on [YOLO-World](https://github.com/AILab-CVC/YOLO-World) and uses its stack: `mmcv==2.0.0`, `mmdet==3.0.0`, `mmengine==0.10.3` and `mmyolo==0.6.0` (included as a submodule).

```bash
git clone --recursive https://github.com/imatif17/DESA-TTA.git
cd DESA-TTA

conda create -n desa-tta python=3.10 -y
conda activate desa-tta

# PyTorch: pick the build that matches your CUDA version
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# mmcv with compiled ops (see the mmcv docs if no pre-built wheel matches your setup)
pip install -U openmim
mim install mmcv==2.0.0

pip install -r requirements/basic_requirements.txt
pip install --no-build-isolation -e third_party/mmyolo
pip install -e .
```

## Data: VOC-C

VOC-C applies the 15 common corruptions of [Michaelis et al. (2019)](https://github.com/bethgelab/imagecorruptions) to the 4,952 images of the PASCAL VOC 2007 test set. To build it:

```bash
# PASCAL VOC 2007 test set
mkdir -p data && cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar   # -> data/VOCdevkit/VOC2007
cd ..

# Corrupted images + COCO-style annotations
pip install -r requirements/voc_c_requirements.txt
python tools/prepare_voc_c.py data/VOCdevkit --out-dir data/VOC-C --severity 5 --workers 8
```

This produces:

```
data/VOC-C/
├── test.json            # VOC 2007 test annotations in COCO format
├── gaussian_noise/      # 4,952 images per corruption
├── shot_noise/
├── ...
└── jpeg_compression/
```

The 20 VOC class embeddings used by YOLO-World are provided in `embeddings/voc_embed.npy`.

## Pretrained detector

Download the YOLO-World-S checkpoint (O365 + GoldG pre-training) from the YOLO-World release:

```bash
mkdir -p pretrained_models
wget -O pretrained_models/yolo_world_s.pth https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_s_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-18bea4d2.pth
```

## Running DESA-TTA

TTA processes the target data as an online stream with batch size 1, so run it on a single GPU. Each corruption is a separate run that starts from the pretrained checkpoint. Run commands from the repository root.

**All 15 corruptions** (prints a summary table at the end):

```bash
bash tools/run_voc_c.sh pretrained_models/yolo_world_s.pth
```

**A single corruption:**

```bash
python tools/test.py configs/voc_c/desa_tta_yolo_world_s_voc_c.py pretrained_models/yolo_world_s.pth \
    --cfg-options test_dataloader.dataset.dataset.data_prefix.img=fog/
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{belal2026desatta,
      title={DESA-TTA: Dynamic EMA and Source Anchoring for Test-Time Adaptation},
      author={Atif Belal and Lilian Hollard and Marco Pedersoli and Eric Granger},
      year={2026},
      eprint={2609.01795},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgments

This code is built on [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [MMYOLO](https://github.com/open-mmlab/mmyolo) and [MMDetection](https://github.com/open-mmlab/mmdetection). VOC-C is generated with [imagecorruptions](https://github.com/bethgelab/imagecorruptions). This work was supported by Distech Controls Inc., the Natural Sciences and Engineering Research Council of Canada (NSERC), the Digital Research Alliance of Canada, and MITACS.

## License

This project is released under the [GPL-3.0 license](LICENSE), following YOLO-World and MMYOLO.
