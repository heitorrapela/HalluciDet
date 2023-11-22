
# HalluciDet (Accepted WACV2024)


![HalluciDet Model](./resources/hallucidet.png)

This repository contains the code used for [HalluciDet: Hallucinating RGB Modality for Person Detection Through Privileged Information 🔗](https://arxiv.org/abs/2310.04662) by Heitor Rapela Medeiros, Fidel A. Guerrero Pena, Masih Aminbeidokhti, Thomas Dubail, Eric Granger, Marco Pedersoli **(WACV 2024)**.



# Dependencies

	conda create -n hallucidet python=3.6
	conda activate hallucidet
	conda install pytorch torchvision torchaudio cpuonly -c pytorch
	pip install matplotlib
	pip install pytorch_lightning
	pip install opencv-python
	pip install imageio
	pip install scikit-image
	pip install scikit-learn
	pip install pandas
	pip install pycocotools
	pip install torchmetrics
	pip install wandb
	pip install lpips
	pip install segmentation-models-pytorch


# How to run


	## For training the initial rgb model that is the baseline (Also you can use our checkpoint, that I am going to update soon)
	## This is the seed123 result for the fasterrcnn on llvip dataset
	CUDA_VISIBLE_DEVICES=0 python train_detector.py --eval --debug --pretrained --threshold 0.5 --wandb-project wacv2024 --wandb-name detector_fasterrcnn_rgb_llvip_200ep_changhead_seed123 --detector fasterrcnn --modality rgb --dataset llvip --epochs 200 --batch 16 --seed 123 -t Paper llvip fasterrcnn rgb
	# Train HalluciDet
	CUDA_VISIBLE_DEVICES=0 python train_encoder.py --eval --debug --pretrained --modality ir --detector-path ./lightning_logs/wacv2024/detector_fasterrcnn_rgb_llvip_200ep_changehead_seed123/llvip_rgb_fasterrcnn/best.ckpt --wandb-project wacv2024 --wandb-name detector_fasterrcnn_hallucidet_det01reg01_llvip_200ep_changehead_seed123 --detector fasterrcnn --dataset llvip --epochs 200 --batch 8 --seed 123 -t Paper llvip fasterrcnn hallucidet

  

# HalluciDet Qualitative Results


![HalluciDet FasterRCNN](./resources/test_batch.gif)


  
# References


Really thanks for the great open source community that provided good libraries