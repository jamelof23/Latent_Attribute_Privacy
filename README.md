# PrivAI - Ensuring Privacy in Latent Space: Differentially Private Image Generation

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 2.0.1+cu117](https://img.shields.io/badge/pytorch-2.0.1%2Bcu117-green.svg?style=plastic)
![CUDA 11.7](https://img.shields.io/badge/cuda-11.7-green.svg?style=plastic)

![image](./docs/image1.jpg)
**Figure:** *Ensuring differential privacy for specific facial attributes while preserving overall image quality results with PrivAI.*

## Introduction
In this repository, we propose

[[Paper](https://arxiv.org/)]
[[Project Page](https://genforce.github.io/interfacegan/)]
[[Colab](https://colab.research.google.com/github/genforce/interfacegan/blob/master/docs/InterFaceGAN.ipynb)]

## Startup Instructions

Pick up a model, pick up a boundary, pick up a latent code, and then EDIT!

```bash
# Before running the following code, please first download
# the pre-trained ProgressiveGAN model on CelebA-HQ dataset,
# and then place it under the folder ".models/pretrain/".
LATENT_CODE_NUM=10
python edit.py \
    -m pggan_celebahq \
    -b boundaries/pggan_celebahq_smile_boundary.npy \
    -n "$LATENT_CODE_NUM" \
    -o results/pggan_celebahq_smile_editing
```

## Prior Work

Before going into details, we would like to first introduce the two state-of-the-art GAN models used in this work, which are ProgressiveGAN (Karras *el al.*, ICLR 2018) and StyleGAN (Karras *et al.*, CVPR 2019). These two models achieve high-quality face synthesis by learning unconditional GANs. For more details about these two models, please refer to the original papers, as well as the official implementations.

ProgressiveGAN:
  [[Paper](https://arxiv.org/pdf/1710.10196.pdf)]
  [[Code](https://github.com/tkarras/progressive_growing_of_gans)]

StyleGAN:
  [[Paper](https://arxiv.org/pdf/1812.04948.pdf)]
  [[Code](https://github.com/NVlabs/stylegan)]


  ## Code Instruction

  ### first code

  ### second code


  ## BibTeX

```bibtex
@inproceedings{shen2020interpreting,
  title     = {Interpreting the Latent Space of GANs for Semantic Face Editing},
  author    = {Shen, Yujun and Gu, Jinjin and Tang, Xiaoou and Zhou, Bolei},
  booktitle = {CVPR},
  year      = {2020}
}
```

```bibtex
@article{shen2020interfacegan,
  title   = {InterFaceGAN: Interpreting the Disentangled Face Representation Learned by GANs},
  author  = {Shen, Yujun and Yang, Ceyuan and Tang, Xiaoou and Zhou, Bolei},
  journal = {TPAMI},
  year    = {2020}
}
```
