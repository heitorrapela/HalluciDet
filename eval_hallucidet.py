import os
from src.config.config import Config
import torch
Config.set_environment()

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from src.utils.utils import Utils
import wandb
import torch.nn as nn

from src.metrics import metrics
import numpy as np
from src.dataloader.dataloaderPL import MultiModalDataModule
import torchvision
from src.models.detector import Detector
from train_detector import DetectorLit
import albumentations as alb
import albumentations.pytorch
from src.models.encoder_decoder import EncoderDecoder

# True = Speed-up but not deterministic
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

args = Config.argument_parser()
seed_everything(args.seed)

dataset = args.dataset if args.dataset is not None else Config.Dataset.dataset
Config.set_dataset_path(dataset)

detector = args.detector if args.detector is not None else Config.Detector.name
Config.set_detector(detector, train_det=False, pretrained=args.directly_coco)

Config.set_loss_weights(args)

ext = args.ext if args.ext is not None else Config.Dataset.ext

wandb_logger = wandb.init(project=args.wandb_project, name=args.wandb_name)

fuse_data = args.fuse_data 

decoder_backbone = args.decoder_backbone

ext = args.ext if args.ext is not None else Config.Dataset.ext

LR = 0.0001 if args.lr is None else args.lr


class EncoderDecoderLit(pl.LightningModule):
    def __init__(self, batch_size=4, 
                wandb_logger=None, model_name='resnet34',
                in_channels=1, output_channels=3,  lr=0.0001,
                loss_pixel='mse', loss_perceptual='lpips_alexnet', 
                detector_name='fasterrcnn', train_det=False, fuse_data='none', scheduler_on=False):
        super().__init__()

        self.model_name = model_name
        self.wandb_logger = wandb_logger
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.lr = lr
        self.batch_size = batch_size
        self.train_det = train_det
        self.fuse_data = fuse_data
        self.optimizer_name = Config.Optimizer.name
        self.segmentation_head = Config.EncoderDecoder.decoder_head
        self.scheduler_on = scheduler_on

        ## EncoderDecoder
        self.encoder_decoder = EncoderDecoder(name=self.model_name, 
                                            encoder_depth=args.encoder_depth,
                                            encoder_weights='imagenet',
                                            decoder_attention_type=None,
                                            in_channels=self.in_channels,
                                            output_channels=self.output_channels,
                                            segmentation_head=Config.EncoderDecoder.decoder_head,
                                            ).encoder_decoder

        ## Detector
        self.detector_name = detector_name

        # Model
        detectorLit = DetectorLit(batch_size=self.batch_size, 
                                wandb_logger=self.wandb_logger,
                                lr=LR, detector_name=self.detector_name, 
                                pretrained=True, optimizer_name=self.optimizer_name, 
                                modality=args.modality,
                                directly_coco=args.directly_coco)
        
        self.detector = detectorLit.detector

        if not Config.Detector.train_det:
            self.detector = self.detector.eval()
            for param in self.detector.parameters():
                param.requires_grad = False

        self.detector = detectorLit.load_from_checkpoint(checkpoint_path=args.detector_path,
                                                batch_size=args.batch, 
                                                wandb_logger=wandb_logger,
                                                lr=LR, 
                                                detector_name=detector_name, 
                                                pretrained=True,
                                                optimizer_name=self.optimizer_name,
                                                modality=args.modality,
                                            ).detector

        ## Metrics
        self.test_metrics_detection_map_hall = metrics.Detection().map
        self.test_metrics_detection_map_rgb = metrics.Detection().map
        self.test_metrics_detection_map_ir = metrics.Detection().map



    def training_step(self, train_batch, batch_idx):

        return None


    def validation_step(self, val_batch, batch_idx):

        return None


    def on_validation_epoch_end(self):
        
        return None

        

    def test_step(self, test_batch, batch_idx):
        
        imgs_rgb, targets_rgb, imgs_ir, targets_ir = test_batch

        imgs_ir = Utils.batch_images_for_encoder_decoder(imgs=imgs_ir, device=device, ablation_flag=args.ablation_flag)
        imgs_rgb = Utils.batch_images_for_encoder_decoder(imgs=imgs_rgb, device=device, ablation_flag=args.ablation_flag)
        
        targets_rgb = Utils.batch_targets_for_detector(targets=targets_rgb, device=device, detector_name=self.detector_name)
        targets_ir = Utils.batch_targets_for_detector(targets=targets_ir, device=device, detector_name=self.detector_name)

        imgs_ir_three_channel = Utils.expand_one_channel_to_output_channels(imgs_ir, self.output_channels)
        imgs_hallucinated = self.encoder_decoder(imgs_ir_three_channel)

        imgs_rgb = imgs_rgb.float() # To handle problems with double to float conversion

        _, detections_hall = Detector.calculate_loss(self.detector, imgs_hallucinated, targets_ir, train_det=False, model_name=self.detector_name)

        ## Detector RGB
        _, detections_rgb  = Detector.calculate_loss(self.detector, imgs_rgb, targets_rgb, train_det=False, model_name=self.detector_name)


        ## Detector IR
        _, detections_ir = Detector.calculate_loss(self.detector, imgs_ir_three_channel, targets_ir, train_det=False, model_name=self.detector_name)

        self.test_metrics_detection_map_rgb.update(detections_rgb, targets_rgb)
        self.test_metrics_detection_map_hall.update(detections_hall, targets_ir)
        self.test_metrics_detection_map_ir.update(detections_ir, targets_ir)


    def on_test_epoch_end(self):

        map_rgb = Utils.filter_dictionary(self.test_metrics_detection_map_rgb.compute(), {'map_50', 'map_75', 'map'})
        map_hall = Utils.filter_dictionary(self.test_metrics_detection_map_hall.compute(), {'map_50', 'map_75', 'map'})
        map_ir = Utils.filter_dictionary(self.test_metrics_detection_map_ir.compute(), {'map_50', 'map_75', 'map'})
    
        self.wandb_logger.summary["test/metrics/map_rgb"] = map_rgb
        self.wandb_logger.summary["test/metrics/map_hall"] = map_hall
        self.wandb_logger.summary["test/metrics/map_ir"] = map_ir

        self.wandb_logger.log({
                        'test/metrics/map_rgb': map_rgb,
                        'test/metrics/map_hall': map_hall,
                        'test/metrics/map_ir': map_ir,
                    })


    def configure_optimizers(self):
        
        optimizer = Config().config_optimizer(optimizer=self.optimizer_name,
                                        params=(list([])),
                                        lr=self.lr)

        return {
            "optimizer": optimizer,
        }

# Set device
device = Config.cuda_or_cpu() if args.device is None else args.device


model = EncoderDecoderLit.load_from_checkpoint(checkpoint_path=args.hallucidet_path,
                                            batch_size=args.batch, 
                                            wandb_logger=wandb_logger,
                                            model_name=decoder_backbone, 
                                            in_channels=Config.EncoderDecoder.in_channels_encoder,
                                            output_channels=Config.EncoderDecoder.out_channels_decoder,
                                            lr=LR,
                                            loss_pixel=Config.Losses.pixel, 
                                            loss_perceptual=Config.Losses.perceptual,
                                            detector_name=Config.Detector.name,
                                            train_det=Config.Detector.train_det,
                                            fuse_data=fuse_data,
                                            scheduler_on=Config.Optimizer.scheduler_on,
                                            strict=False
                                        )


# Training
trainer = pl.Trainer(
                    gpus=Config.Environment.N_GPUS,
                    accelerator="gpu",
                    max_epochs=args.epochs,
                    gradient_clip_val=Config.Optimizer.gradient_clip_val, 
                    gradient_clip_algorithm="value",
                    callbacks=[
                            pl.callbacks.RichProgressBar(),
                    ],
                    deterministic=True,
                    limit_train_batches=0.03,
                    limit_val_batches=0.03,
                    num_sanity_val_steps=0,
                    precision=args.precision, # 32 default
                    enable_model_summary=True,
                    logger=False,
                    )

# Fixed transformations
fixed_transformations = alb.Compose(
    [
    alb.pytorch.ToTensorV2(),
    ]
)

# data augmentation
data_augmentation = alb.Compose(
    [fixed_transformations],  
    bbox_params=alb.BboxParams(format='pascal_voc', label_fields=['labels']), 
    additional_targets={'image1': 'image', 'bboxes1': 'bboxes', 'labels1': 'labels'}, p=1.0
 )

dm = MultiModalDataModule(
                        dataset=dataset,
                        path_images_train_rgb=Config.Dataset.train_path,
                        path_images_train_ir=Config.Dataset.train_path,
                        path_images_test_rgb=Config.Dataset.test_path, 
                        path_images_test_ir=Config.Dataset.test_path,
                        batch_size=args.batch, 
                        num_workers=args.num_workers, 
                        ext=ext,
                        seed=args.seed,
                        split_ratio_train_valid=Config.Dataset.train_valid_split,
                        data_augmentation=data_augmentation,
                        fixed_transformations=fixed_transformations,
                        ablation_flag=args.ablation_flag,
                        )

trainer.fit(model, dm)

trainer.test(model, dm, ckpt_path="best")

wandb_logger.summary["checkpoint_dirpath"] = trainer.checkpoint_callback.dirpath

wandb_logger.finish()