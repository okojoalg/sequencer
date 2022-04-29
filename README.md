**Sequencer**: Deep LSTM for Image Classification
========

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg?style=plastic)](https://arxiv.org/abs/xxxx.xxxxx)
[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=plastic&labelColor=005BBB)](https://opensource.fb.com/support-ukraine)

Created by
* [Yuki Tatsunami](https://sites.google.com/view/yuki-tatsunami) 
  * [![Rikkyo University](https://img.shields.io/badge/Rikkyo-University-FFFFFF?style=plastic&labelColor=582780)](https://www.rikkyo.ac.jp)
  * [![AnyTech](https://img.shields.io/badge/AnyTech-Co.%20Ltd.-18C4AA?style=plastic&labelColor=254BB1)](https://anytech.co.jp/)
* [Masato Taki](https://scholar.google.com/citations?hl=en&user=3nMhvfgAAAAJ)
  * [![Rikkyo University](https://img.shields.io/badge/Rikkyo-University-FFFFFF?style=plastic&labelColor=582780)](https://www.rikkyo.ac.jp)

This repository contains implementation for Sequencer.

## Abstract


## Schematic diagrams

The overall architecture of Sequencer2D is similar to the typical hierarchical ViT and Visual MLP. It uses Sequencer2D blocks instead of Transformer blocks:

![Sequencer]

Sequencer2D block replaces the Transformer's self-attention layer with an LSTM-based layer like BiLSTM2D layer:

![Sequencer2D]

BiLSTM2D includes a vertical LSTM and a horizontal LSTM:

![BiLSTM2D]

[Sequencer]: img/Sequencer.png
[Sequencer2D]: img/Sequencer2D.png
[BiLSTM2D]: img/BiLSTM2D.png

## Model Zoo
We provide our Sequencer models pretrained on ImageNet-1K:
| name | arch | Params | FLOPs | acc@1 | download |
| --- | --- | --- | --- | --- | --- |
| Sequencer2D-S | ```sequencer2d_s``` | 28M | 8.4G | 82.3 | [here](https://github.com/okojoalg/sequencer/releases/download/weights/sequencer2d_s.pth) |
| Sequencer2D-M | ```sequencer2d_m``` | 38M | 11.1G | 82.8 | [here](https://github.com/okojoalg/sequencer/releases/download/weights/sequencer2d_m.pth) |
| Sequencer2D-L | ```sequencer2d_l``` | 54M | 16.6G | 83.4 | [here](https://github.com/okojoalg/sequencer/releases/download/weights/sequencer2d_l.pth) |

## Usage

### Requirements
- torch>=1.10.0
- torchvision
- timm==0.5.4
- Pillow
- matplotlib
- scipy
- etc., see [requirements.txt](requirements.txt)

### Data preparation
Download and extract ImageNet images. The directory structure should be as follows.

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Traning
Command line for training Sequencer models on ImageNet from scratch.
```
./distributed_train.sh 8 /path/to/imagenet --model sequencer2d_s -b 256 -j 8 --opt adamw --epochs 300 --sched cosine --native-amp --img-size 224 --drop-path 0.1 --lr 2e-3 --weight-decay 0.05 --remode pixel --reprob 0.25 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --warmup-lr 1e-6 --warmup-epochs 20
```

Command line for fine-tuning a pre-trained model at higher resolution.
```
./distributed_train.sh 8 /path/to/imagenet --model sequencer2d_l --pretrained -b 64 -j 8 --opt adamw --epochs 30 --sched cosine --native-amp --input-size 3 392 392 --img-size 392 --crop-pct 1.0 --drop-path 0.4 --lr 5e-5 --weight-decay 1e-8 --remode pixel --reprob 0.25 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --warmup-epochs 0 --cooldown-epochs 0
```

Command line for fine-tuning a pre-trained model on a transfer learning dataset.
```
./distributed_train.sh 4 /path/to/cifar10 --model sequencer2d_s -b 128 -j 4 --num-classes 10 --dataset torch/cifar10 --pretrained --opt adamw --epochs 200 --sched cosine --native-amp --img-size 224 --clip-grad 1 --drop-path 0.1 --lr 0.0001 --weight-decay 1e-4 --remode pixel --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --warmup-lr 1e-6 --warmup-epochs 5
```

### Validation
To evaluate our Sequencer models, run:
```
python validate.py /path/to/imagenet --model sequencer2d_s -b 16 --input-size 3 224 224 --amp
```

## Reference
You may want to cite:
```

```

## Acknowledgment
This implementation is based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) by Ross Wightman. We thank for his brilliant work.

|   |   |
|:--|:-:|
|  We thank [Graduate School of Artificial Intelligence and Science, Rikkyo University (Rikkyo AI)](https://ai.rikkyo.ac.jp) which supports us with computational resources, facilities, and others. |  ![logo-rikkyo-ai] |
|  [AnyTech Co. Ltd.](https://anytech.co.jp) provided valuable comments on the early versions and encouragement. We thank them for their cooperation. In particular, We thank Atsushi Fukuda for organizing discussion opportunities. |  ![logo-anytech] |

[logo-rikkyo-ai]: img/RIKKYOAI_main.png "Logo of Rikkyo AI"
[logo-anytech]: img/anytech.svg "Logo of AnyTech"