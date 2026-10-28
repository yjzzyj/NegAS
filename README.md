# NegAS

Official PyTorch implementation of **NegAS: Negative Label Guided Attention and Scoring for Out-of-Distribution Object Detection with Vision-Language Models** (ECCV 2026).

**Yingjie Zhang, Shuai Li, Peng Wang**  
Northwestern Polytechnical University

[Paper](https://arxiv.org/abs/2606.22537) · [Overview](#overview) · [Main Results](#main-results) · [Installation](#installation) · [Data Preparation](#data-preparation) · [Training](#training) · [Inference](#inference) · [Evaluation](#evaluation)

## Overview

Vision-language object detectors can confidently mistake unseen objects for known categories. Their ID-label-guided attention treats background regions uniformly, overlooking potential out-of-distribution (OOD) evidence, while their sigmoid-based classification outputs call for an OOD score tailored to this formulation. **NegAS uses negative labels to improve the separation between in-distribution (ID) and OOD objects during both training and scoring.**

Negative labels describe categories that are **visually similar to, but semantically distinct from, ID categories**. We generate candidates with an LLM offline, filter them by CLIP visual similarity to ID objects, and select labels with high semantic dissimilarity to the ID vocabulary. **Negative Label Guided Attention (NegA)** uses these labels in an auxiliary training branch, with supervision restricted to background regions using masks constructed from ground-truth boxes. In the YOLO-World implementation, the detector parameters remain frozen and only shared text prompts are optimized. The auxiliary branch is discarded at inference, so NegA adds no inference overhead in the neck.

**Negative Label Guided Scoring (NegS)** uses the remaining ID branch to compare each detection against both ID and negative labels. Its score is the maximum ID sigmoid score minus the maximum negative-label sigmoid score; higher values indicate stronger ID evidence. This retains negative-label information for OOD scoring without running the auxiliary attention branch at test time.

This repository provides the YOLO-World implementation, including prompt training, object detection, OOD evaluation, and curated negative labels for PASCAL VOC and BDD100K. Negative-label generation is performed offline; training and inference require no LLM calls.

## Main Results

The following results are reported in **Table 1 of the [paper](https://arxiv.org/abs/2606.22537)**. PASCAL VOC and BDD100K serve as ID datasets; MS-COCO and OpenImages serve as OOD datasets with overlapping ID categories removed. All values are percentages: lower **FPR95** is better, while higher **AUROC** and **ID mAP** are better. These are paper-reported results, rather than new measurements from this release.

### YOLO-World

| ID dataset | Method | MS-COCO FPR95 ↓ | MS-COCO AUROC ↑ | OpenImages FPR95 ↓ | OpenImages AUROC ↑ | ID mAP ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VOC | YOLO-World | 21.3 | 93.4 | 48.8 | 81.4 | 69.0 |
| VOC | YOLO-World + CoOp | 13.9 | 95.7 | 47.0 | 83.3 | 72.3 |
| VOC | **YOLO-World + NegAS** | **9.9** | **95.9** | **23.3** | **92.6** | **72.5** |
| BDD100K | YOLO-World | 40.4 | 83.9 | 8.2 | 97.0 | 22.5 |
| BDD100K | YOLO-World + CoOp | 47.0 | 82.0 | 15.1 | 92.6 | 26.1 |
| BDD100K | **YOLO-World + NegAS** | **20.9** | **92.0** | **3.6** | **98.7** | **29.0** |

With VOC as ID, NegAS reduces FPR95 from **21.3 to 9.9** on MS-COCO and from **48.8 to 23.3** on OpenImages, improvements of **11.4 and 25.5 percentage points** over the original YOLO-World baseline. CoOp denotes ID-only prompt tuning. As discussed in the paper, much of the VOC ID mAP gain over the original detector comes from prompt tuning; NegAS preserves this ID accuracy while substantially improving OOD detection.


See the paper for comparisons with other OOD detectors, scoring-function comparisons, and ablation studies.

## Installation

Use Python 3.10 or later. Install a matching PyTorch and torchvision pair for your CUDA environment, then run the following command from the repository root:

```bash
python -m pip install -e .
```

Obtain the `yolov8x-worldv2.pt` pretrained checkpoint from the [YOLO-World model page](https://docs.ultralytics.com/models/yolo-world/) and place it in `weights/`:

```text
weights/
└── yolov8x-worldv2.pt
```

The CLIP weights are downloaded automatically on first use. Datasets and model checkpoints are not bundled with this repository.

## Data Preparation

We use **PASCAL VOC** and **BDD100K** as ID datasets, and **MS-COCO** and **OpenImages** as OOD datasets. Prepare OOD splits with the ID categories excluded, following the evaluation protocol in the paper.

### Download links (VOS)

We follow the dataset preparation protocol of [VOS (ICLR 2022)](https://github.com/deeplearning-wisc/vos#dataset-preparation). The links below are provided by the VOS authors.

| ID dataset | Download |
| --- | --- |
| PASCAL VOC 2007+2012 | [Processed images and annotations](https://drive.google.com/file/d/1n9C4CiBURMSCZy2LStBQTzR17rD_a67e/view?usp=sharing) |
| BDD100K | [Images](https://bdd-data.berkeley.edu/) · [Training annotations](https://drive.google.com/file/d/1ZbbdKEakSjyOci7Ggm046hCCGYqIHcbE/view?usp=sharing) · [Validation annotations](https://drive.google.com/file/d/1Rxb9-6BUUGZ_VsNZy9S2pWM8Q5goxrXY/view?usp=sharing) |

VOS provides the following OOD image subsets for each ID dataset. Download the subsets matching your ID setting:

| ID setting | MS-COCO OOD images | OpenImages OOD images |
| --- | --- | --- |
| VOC | [ID-VOC-OOD-COCO](https://drive.google.com/drive/folders/1NxodhoxTX5YBHJWHAa6tB2Ta1oxoTfzu?usp=sharing) | [ID-VOC-OOD-OpenImages](https://drive.google.com/drive/folders/1pRP7CAWG7naDECfejo03cl7PF3VJEjrn?usp=sharing) |
| BDD100K | [ID-BDD-OOD-COCO](https://drive.google.com/drive/folders/1Wgmfcp2Gd3YvYVyoRRBWUiwwKYXJeuo8?usp=sharing) | [ID-BDD-OOD-OpenImages](https://drive.google.com/drive/folders/1LyOFqSm2G8x7d2xUkXma2pFJVOLgm3IQ?usp=sharing) |

For preparation from the original COCO2017 images, use VOS's processed OOD annotations for [VOC as ID](https://drive.google.com/file/d/1Wsg9yBcrTt2UlgBcf7lMKCw19fPXpESF/view?usp=sharing) or [BDD100K as ID](https://drive.google.com/file/d/1AOYAJC5Z5NzrLl5IIJbZD4bbrZpo0XPh/view?usp=sharing). VOS also provides an [OpenImages validation archive](https://drive.google.com/file/d/1UPuxoE1ZqCfCZX48H7bWX7GGIJsTUrt5/view?usp=sharing).

The processed ID annotations are JSON files. Convert them to YOLO format, remap category IDs to the order in our dataset configuration, and preserve the supplied train/validation split. OOD image subsets can be passed directly to `predict.py`; OOD scoring does not require converting their annotations.

### ID datasets

Convert the annotations to YOLO detection format and organize each ID dataset as follows:

```text
datasets/
├── voc/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
└── bdd/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
```

Each image has a matching `.txt` annotation file. Each annotation row contains `class_id x_center y_center width height`, with zero-based class indices and box coordinates normalized to `[0, 1]`.

Update the `path` field in the corresponding dataset configuration to your local dataset root:

| ID dataset | ID classes | Dataset configuration | Class names |
| --- | ---: | --- | --- |
| PASCAL VOC | 20 | [`data/voc_neg20.yaml`](data/voc_neg20.yaml) | [`data/voc_neg20.names`](data/voc_neg20.names) |
| BDD100K | 10 | [`data/bdd_neg20.yaml`](data/bdd_neg20.yaml) | [`data/bdd_neg20.names`](data/bdd_neg20.names) |

The YAML and `.names` files must use the same class order: **ID classes first, followed by 20 negative labels**. Training annotations contain only ID classes and must match this ordering. Negative labels are text categories and require no additional bounding-box annotations.

### OOD datasets

Store the prepared OOD images in separate directories, for example:

```text
datasets/
├── coco_ood/images/
└── openimages_ood/images/
```

OOD scoring uses detections exported from these images. Use the vocabulary associated with the trained ID dataset for both ID and OOD inference.

## Training

The default setting uses AdamW with a learning rate of `0.01`, `30` epochs, an image size of `640`, and a total batch size of `160`. Only the shared text prompts are optimized.

For PASCAL VOC:

```bash
python train.py \
  --data data/voc_neg20.yaml \
  --names data/voc_neg20.names \
  --n_id_cls 20 \
  --weight weights/yolov8x-worldv2.pt \
  --device 0,1,2,3 \
  --batch 160 \
  --dataset voc \
  --project runs
```

For BDD100K:

```bash
python train.py \
  --data data/bdd_neg20.yaml \
  --names data/bdd_neg20.names \
  --n_id_cls 10 \
  --weight weights/yolov8x-worldv2.pt \
  --device 0,1,2,3 \
  --batch 160 \
  --dataset bdd \
  --project runs
```

`--batch` is the total batch size across all devices. Adjust `--device` and `--batch` to fit your hardware. Checkpoints are saved under the run directory, such as `runs/voc/train/weights/`; repeated runs receive an incremented directory name.

To resume training, use the same dataset arguments with `--resume --weight /path/to/last.pt`.

## Inference

Run detection on an image or image directory with a trained NegAS checkpoint:

```bash
python predict.py \
  --weight runs/voc/train/weights/best.pt \
  --source /path/to/images \
  --names data/voc_neg20.names \
  --n-id-cls 20 \
  --device 0 \
  --dataset demo \
  --save
```

The script saves `predictions.json` and prints its location. Each detection includes its box, confidence, class index, and ID/negative-label logits. Add `--save` to save visualized detections. For a BDD100K model, use `data/bdd_neg20.names` and `--n-id-cls 10`.

## Evaluation

### ID detection performance

Evaluate ID detection accuracy on the validation split:

```bash
python val.py \
  --weight runs/voc/train/weights/best.pt \
  --data data/voc_neg20.yaml \
  --names data/voc_neg20.names \
  --device 0 \
  --batch 20 \
  --dataset voc \
  --save-json
```

For BDD100K, replace the dataset configuration and class-name file with the corresponding BDD100K files.

### OOD detection performance

First export predictions for the ID validation images and an OOD image set using the same model and class vocabulary:

```bash
# ID predictions
python predict.py \
  --weight runs/voc/train/weights/best.pt \
  --source /path/to/voc/val/images \
  --names data/voc_neg20.names \
  --n-id-cls 20 \
  --device 0 \
  --dataset voc_id

# OOD predictions (repeat with OpenImages to evaluate that split)
python predict.py \
  --weight runs/voc/train/weights/best.pt \
  --source /path/to/coco_ood/images \
  --names data/voc_neg20.names \
  --n-id-cls 20 \
  --device 0 \
  --dataset coco_ood
```

Then compute NegS and the OOD detection metrics from the exported JSON files:

```bash
python eval_ood.py \
  --id-json /path/to/id/predictions.json \
  --ood-json /path/to/ood/predictions.json \
  --n-id-cls 20 \
  --id-conf 0.5 \
  --output output/voc_coco_metrics.json
```

Select `--id-conf` on the ID validation set; `0.5` above is an example. The same threshold is applied to OOD predictions by default. Keep the prediction export threshold (`--conf`, default `0.001`) below the evaluation threshold.

The evaluator reports **FPR95**, **AUROC**, and **AUPR** as values in `[0, 1]`. NegS is the maximum ID sigmoid score minus the maximum negative-label sigmoid score; a larger value indicates a stronger ID prediction.

## Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{NegAS,
  author={Zhang, Yingjie and Li, Shuai and Wang, Peng},
  title={NegAS: Negative Label Guided Attention and Scoring for Out-of-Distribution Object Detection with Vision-Language Models},
  booktitle={European conference on computer vision},
  year="2026",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="1--18",
  isbn="978-3-032-37258-1"
}

```

## Acknowledgements

We thank [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [Ultralytics](https://github.com/ultralytics/ultralytics), and [CLIP](https://github.com/openai/CLIP) for their open-source contributions. This repository is distributed under the [AGPL-3.0 license](LICENSE).
