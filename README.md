
# PCNet-M Experiments on COCOA Dataset

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

## Contents
1. [Overview](#1-overview)
2. [Setup Instructions](#2-setup-instructions)
3. [Experiments](#3-experiments)

## 1. Overview

This repo contains the code for my experiments on **mask completion** using the PCNet-M model proposed in [Self-Supervised Scene De-occlusion](https://xiaohangzhan.github.io/projects/deocclusion/).

## 2. Setup Instructions

- Clone the repo:

```shell
git clone https://github.com/praeclarumjj3/PCNetM-Experiments.git
cd PCNetM-Experiments
```

- Install pycocotools:
   
```shell
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
```

- Install [Pytorch](https://pytorch.org/get-started/locally/) and other dependencies:

```shell
pip3 install -r requirements.txt
```

### Dataset Preparation

- Download the **MS-COCO 2014** images and unzip:
```
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```

- Download the annotations and untar:
``` 
gdown https://drive.google.com/uc?id=0B8e3LNo7STslZURoTzhhMFpCelE
tar -xf annotations.tar.gz
```

- Unzip the files according to the following structure

```
PCNetM-Experiments
├── data
│   ├── COCOA
│   │   ├── annotations
│   │   ├── train2014
│   │   ├── val2014
```

### Run Demos

1. Download released models [here](https://drive.google.com/drive/folders/1O89ItVWucCoL_VxIbLM1XLxr9JFfyj_Y?usp=sharing) and put the folder `released` under `PCNetM-Experiments`.

2. Run `demos/demo_cocoa.ipynb`. There are some test examples for `demos/demo_cocoa.ipynb` in the repo, so you don't have to download the COCOA dataset if you just want to try a few samples.

3. If you want to use predicted modal masks by existing instance segmentation models, you need to adjust some parameters in the demo, please refer to the answers in this [issue](https://github.com/XiaohangZhan/deocclusion/issues/14).

## 3. Experiments

### Training

- Run the following command:

```
sh experiments/COCOA/pcnet_m/train.sh # you may have to set--nproc_per_node=#YOUR_GPUS
```

2. Monitoring status and visual results using tensorboard (if `True` in config.yaml):

```
sh tensorboard.sh $PORT
```

## Evaluate

- Execute:

```shell
sh tools/test_cocoa.sh
```

## Acknowledgement

This repo borrows heavily from [deocclusion](https://github.com/XiaohangZhan/deocclusion).
