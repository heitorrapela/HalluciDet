# [WACV2024] HalluciDet: Hallucinating RGB Modality for Person Detection Through Privileged Information

[![Paper](https://img.shields.io/badge/Paper-WACV2024-blue)](https://openaccess.thecvf.com/content/WACV2024/html/Medeiros_HalluciDet_Hallucinating_RGB_Modality_for_Person_Detection_Through_Privileged_Information_WACV_2024_paper.html)
[![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-orange)](https://huggingface.co/heitorrapela/hallucidet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---

**Abstract:**  
HalluciDet is a novel framework for person detection in thermal images by hallucinating the missing RGB modality at test time. Leveraging privileged information during training, HalluciDet significantly boosts detection accuracy in challenging scenarios where only infrared data is available. Our method outperforms previous approaches on the LLVIP dataset, achieving a new state-of-the-art.

**Key Contributions:**
- Hallucinates RGB features from IR input for improved detection.
- Outperforms standard and domain adaptation baselines on LLVIP.
- Easy-to-use PyTorch code and pretrained models available.

---

Recently, this work was also accepted as an extended abstract in the [LatinX in CV (LXCV) @CVPR2024 🔗](https://www.latinxinai.org/cvpr-2024)

![HalluciDet Model](./resources/hallucidet.png)

---

## Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/yourusername/HalluciDet.git
cd HalluciDet
conda create -n hallucidet python=3.8.10
conda activate hallucidet

# Install dependencies
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torchmetrics==0.6.0 matplotlib==3.7.0 pytorch-lightning==1.5.10.post0 opencv-python==4.6.0.66 imageio==2.22.4 scikit-image==0.19.3 scikit-learn==1.1.3 pandas==1.5.3 pycocotools wandb==0.19.5 albumentations==1.3.1 timm==0.6.12 pretrainedmodels==0.7.4 rich

# Download pretrained weights
git lfs install
git clone https://huggingface.co/heitorrapela/hallucidet
ln -s hallucidet/checkpoints/ .

# Run inference on a sample image
python eval_hallucidet.py --pretrained --modality ir --detector-path ./checkpoints/llvip/seed123/fasterrcnn_rgb_llvip_seed123.ckpt --hallucidet-path ./checkpoints/llvip/seed123/hallucidet_llvip_seed123.ckpt --dataset llvip --epochs 1 --batch 1 --seed 123
```

---

## Why Use HalluciDet?

- **State-of-the-art**: Achieves AP@50=90.57 on LLVIP (IR input).
- **Plug-and-play**: Works with your IR data, no RGB needed at test time.
- **Open-source**: Easy to extend for new datasets and detectors.

---

## HalluciDet Qualitative Results

![HalluciDet FasterRCNN](./resources/test_batch.gif)

---

## Talks about this work

**WACV2024 Recorded Video**  
[![WACV2024 Recorded Video](https://img.youtube.com/vi/BEFi_zkG8Yc/0.jpg)](https://www.youtube.com/watch?v=BEFi_zkG8Yc)

**Talk at [LIVIA](https://liviamtl.ca/)**  
[![Talk at LIVIA](https://img.youtube.com/vi/spH6mHMHapw/0.jpg)](https://youtu.be/spH6mHMHapw)

---

## Dependencies

    conda create -n hallucidet python=3.8.10
    conda activate hallucidet

    # I recommend installing each one manually; cu113 has some problems with pip in the requirements.txt
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install torchmetrics==0.6.0
    pip install matplotlib==3.7.0
    pip install pytorch-lightning==1.5.10.post0
    pip install opencv-python==4.6.0.66
    pip install imageio==2.22.4
    pip install scikit-image==0.19.3
    pip install scikit-learn==1.1.3
    pip install pandas==1.5.3
    pip install pycocotools
    pip install wandb==0.19.5
    pip install albumentations==1.3.1
    pip install timm==0.6.12
    pip install pretrainedmodels==0.7.4
    pip install rich

---

## Dataset preparation

Place the dataset in the same folder as the code for HalluciDet.

    # If you already have the dataset, you can just create a symbolic link, e.g., ln -s ./datasets/LLVIP
    . root
    ├── datasets
    └── HalluciDet

    cd ..
    mkdir datasets
    ln -s ./datasets/LLVIP 
    cd HalluciDet

---

## How to Train (Detectors/HalluciDet)

    ## For training the initial rgb model that is the baseline
    CUDA_VISIBLE_DEVICES=0 python train_detector.py --pretrained --wandb-project wacv2024 --wandb-name detector_fasterrcnn_rgb_llvip_200ep_seed123 --detector fasterrcnn --modality rgb --dataset llvip --epochs 200 --batch 16 --seed 123

    ## Train HalluciDet (Check if you are loading the correct path for the detector)
    CUDA_VISIBLE_DEVICES=0 python train_hallucidet.py --pretrained --modality ir --detector-path ./lightning_logs/wacv2024/detector_fasterrcnn_rgb_llvip_200ep_seed123/llvip_rgb_fasterrcnn/best.ckpt --wandb-project wacv2024 --wandb-name detector_fasterrcnn_hallucidet_det01reg01_llvip_200ep_seed123 --detector fasterrcnn --dataset llvip --epochs 200 --batch 8 --seed 123

---

## How to Eval (HalluciDet)

Download the pre-trained weights: https://huggingface.co/heitorrapela/hallucidet/tree/main (update the --detector-path and --hallucidet-path)

    # You can download the weights manually or you can use git-lfs
    git lfs install
    git clone https://huggingface.co/heitorrapela/hallucidet
    ln -s hallucidet/checkpoints/ .

    ## Eval for Faster R-CNN HalluciDet
    CUDA_VISIBLE_DEVICES=0 python eval_hallucidet.py --pretrained --modality ir --detector-path ./checkpoints/llvip/seed123/fasterrcnn_rgb_llvip_seed123.ckpt --hallucidet-path ./checkpoints/llvip/seed123/hallucidet_llvip_seed123.ckpt --wandb-project wacv2024 --wandb-name detector_fasterrcnn_hallucidet_det01reg01_llvip_200ep_seed123 --detector fasterrcnn --dataset llvip --epochs 1 --batch 8 --seed 123

    # You should get something like:
    RGB Detector on IR  AP@50:  69.75
    RGB Detector on RGB AP@50:  76.86
    HalluciDet   on IR  AP@50:  90.57

---

## Citation

If you use HalluciDet, please cite:

```bibtex
@inproceedings{medeiros2024hallucidet,
  title={HalluciDet: Hallucinating RGB Modality for Person Detection Through Privileged Information},
  author={Medeiros, Heitor Rapela and Pena, Fidel A Guerrero and Aminbeidokhti, Masih and Dubail, Thomas and Granger, Eric and Pedersoli, Marco},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1444--1453},
  year={2024}
}
```

You may also be interested in our follow-up works:

```bibtex
@inproceedings{medeiros2024modality,
  title={Modality translation for object detection adaptation without forgetting prior knowledge},
  author={Medeiros, Heitor Rapela and Aminbeidokhti, Masih and Pe{\~n}a, Fidel Alejandro Guerrero and Latortue, David and Granger, Eric and Pedersoli, Marco},
  booktitle={European Conference on Computer Vision},
  pages={51--68},
  year={2024},
  organization={Springer}
}

@article{medeiros2024visual,
  title={Visual Modality Prompt for Adapting Vision-Language Object Detectors},
  author={Medeiros, Heitor R and Belal, Atif and Muralidharan, Srikanth and Granger, Eric and Pedersoli, Marco},
  journal={arXiv preprint arXiv:2412.00622},
  year={2024}
}
```

---

## References

Thanks to the great open-source community that provided good