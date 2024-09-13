import os
from src.config.config import Config
import torch
Config.set_environment()

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from src.utils.utils import Utils
import wandb
import torch.nn as nn

from src.losses import losses
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

wandb_logger = wandb.init(project=args.wandb_project, name=args.wandb_name, tags=args.tags)

fuse_data = args.fuse_data 

decoder_backbone = args.decoder_backbone

ext = args.ext if args.ext is not None else Config.Dataset.ext

pre_train_path = None
if Config.EncoderDecoder.load_encoder_decoder:
    pre_train_path = args.pre_train_path if args.pre_train_path is not None else Config.EncoderDecoder.encoder_decoder_load_path

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

        self.loss_pixel = losses.Reconstruction.select_loss_pixel(loss_pixel=loss_pixel)
        self.loss_perceptual = losses.Reconstruction.select_loss_perceptual(loss_perceptual=loss_perceptual)

        ## Metrics
        self.train_metrics_detection_map_hall = metrics.Detection().map
        self.valid_metrics_detection_map_hall = metrics.Detection().map
        self.test_metrics_detection_map_hall = metrics.Detection().map

        self.train_metrics_detection_map_rgb = metrics.Detection().map
        self.valid_metrics_detection_map_rgb = metrics.Detection().map
        self.test_metrics_detection_map_rgb = metrics.Detection().map

        self.train_metrics_detection_map_ir = metrics.Detection().map
        self.valid_metrics_detection_map_ir = metrics.Detection().map
        self.test_metrics_detection_map_ir = metrics.Detection().map

        self.train_epoch = 0
        self.valid_epoch = 0

        self.train_media_step = 0
        self.valid_media_step = 0

        self.train_loss_step = 0
        self.valid_loss_step = 0

        self.best_valid_map_50 = 0.0
        self.best_valid_epoch = 0

        self.wandb_logger.define_metric("train/loss/step")
        self.wandb_logger.define_metric("train/loss/*", step_metric="train/loss/step")

        self.wandb_logger.define_metric("train/media/step")
        self.wandb_logger.define_metric("train/media/*", step_metric="train/media/step")

        self.wandb_logger.define_metric("valid/loss/step")
        self.wandb_logger.define_metric("valid/loss/*", step_metric="valid/loss/step")

        self.wandb_logger.define_metric("valid/media/step")
        self.wandb_logger.define_metric("valid/media/*", step_metric="valid/media/step")

        self.wandb_logger.define_metric("valid/metrics/step")
        self.wandb_logger.define_metric("valid/metrics/*", step_metric="valid/metrics/step")
        

    def forward_step(self, imgs_rgb, targets_rgb, imgs_ir, targets_ir, batch_idx, step='train'):

        imgs_ir = Utils.batch_images_for_encoder_decoder(imgs=imgs_ir, device=device, ablation_flag=args.ablation_flag)
        imgs_rgb = Utils.batch_images_for_encoder_decoder(imgs=imgs_rgb, device=device, ablation_flag=args.ablation_flag)

        targets_rgb = Utils.batch_targets_for_detector(targets=targets_rgb, device=device, detector_name=self.detector_name)
        targets_ir = Utils.batch_targets_for_detector(targets=targets_ir, device=device, detector_name=self.detector_name)
        
        ## Encoder / Decoder
        imgs_ir_three_channel = Utils.expand_one_channel_to_output_channels(imgs_ir, self.output_channels)
        imgs_hallucinated = self.encoder_decoder(imgs_ir_three_channel)

        loss_pixel_rgb = 0.0 if self.loss_pixel == None else self.loss_pixel(imgs_rgb, imgs_hallucinated) * Config.Losses.hparams_losses_weights['pixel_rgb']
        loss_perceptual_rgb = 0.0 if self.loss_perceptual == None else torch.mean(self.loss_perceptual(imgs_rgb, imgs_hallucinated)) * Config.Losses.hparams_losses_weights['perceptual_rgb']
        loss_pixel_ir = 0.0 if self.loss_pixel == None else self.loss_pixel(imgs_ir_three_channel, imgs_hallucinated) * Config.Losses.hparams_losses_weights['pixel_ir']
        loss_perceptual_ir = 0.0 if self.loss_perceptual == None else torch.mean(self.loss_perceptual(imgs_ir_three_channel, imgs_hallucinated)) * Config.Losses.hparams_losses_weights['perceptual_ir']
        
        ## Detector Hallucinated
        train_det = True if (self.train_det == True and step == 'train') else False
        losses_det, detections_hall = Detector.calculate_loss(self.detector, imgs_hallucinated, targets_ir, train_det=train_det, model_name=self.detector_name)

        ## Detector RGB
        _, detections_rgb  = Detector.calculate_loss(self.detector, imgs_rgb, targets_rgb, train_det=False, model_name=self.detector_name)

        ## Detector IR
        _, detections_ir = Detector.calculate_loss(self.detector, imgs_ir_three_channel, targets_ir, train_det=False, model_name=self.detector_name)


        if 'fasterrcnn' in self.detector_name:
            losses_det['classification'] = losses_det['loss_classifier']
            losses_det['bbox_regression'] = losses_det['loss_box_reg']

        losses_det['bbox_regression'] = losses_det['bbox_regression'] * Config.Losses.hparams_losses_weights['det_regression']
        losses_det['classification'] = losses_det['classification'] * Config.Losses.hparams_losses_weights['det_classification']
        
        losses_det['loss_objectness'] = (losses_det['loss_objectness'] *  Config.Losses.hparams_losses_weights['det_objectness']
                                            if 'fasterrcnn' in self.detector_name else 0.0)
        losses_det['loss_rpn_box_reg'] = (losses_det['loss_rpn_box_reg'] * Config.Losses.hparams_losses_weights['det_rpn_box_reg']
                                            if 'fasterrcnn' in self.detector_name else 0.0)
        losses_det['bbox_ctrness'] = (losses_det['bbox_ctrness'] * Config.Losses.hparams_losses_weights['det_bbox_ctrness'] 
                                            if 'fcos' in self.detector_name else 0.0)
        
        loss_det_total = losses_det['bbox_regression'] + losses_det['classification'] + \
                        losses_det['loss_objectness'] + \
                        losses_det['loss_rpn_box_reg'] + losses_det['bbox_ctrness']                               


        ## Total Loss
        total_loss = loss_det_total + loss_pixel_rgb + loss_perceptual_rgb + loss_pixel_ir + loss_perceptual_ir

        if step == 'val':

            self.valid_metrics_detection_map_rgb.update(detections_rgb, targets_rgb)
            self.valid_metrics_detection_map_hall.update(detections_hall, targets_ir)
            self.valid_metrics_detection_map_ir.update(detections_ir, targets_ir)

        # Normalize for plotting
        imgs_hallucinated = Utils.normalize_batch_images(imgs_hallucinated.detach().clone())

        return {
            'loss' : {'total': total_loss, 
                'pixel_rgb': loss_pixel_rgb, 
                'perceptual_rgb': loss_perceptual_rgb, 
                'pixel_ir': loss_pixel_ir,
                'perceptual_ir': loss_perceptual_ir, 
                'det_regression': losses_det['bbox_regression'],
                'det_classification': losses_det['classification'],
                'det_objectness': losses_det['loss_objectness'],
                'det_rpn_box_reg': losses_det['loss_rpn_box_reg'],
                'det_bbox_ctrness': losses_det['bbox_ctrness'],
                'det_total': loss_det_total,
                },
            'output': { #'det_hal': output_hal_det, 
                        # 'det_rgb': output_rgb_det, 
                        # 'det_ir': output_ir_det,
                        'imgs_rgb': imgs_rgb,
                        'imgs_ir': imgs_ir,
                        'imgs_hallucinated': imgs_hallucinated,
                    }
        }


    def training_step(self, train_batch, batch_idx):

        imgs_rgb, targets_rgb, imgs_ir, targets_ir = train_batch

        forward_return =  self.forward_step(imgs_rgb, targets_rgb, imgs_ir, targets_ir, batch_idx, step='train')
        
        self.log('train_loss', forward_return['loss']['total'], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.wandb_logger.log({ 'train/loss/pixel_rgb': forward_return['loss']['pixel_rgb'], 
                                'train/loss/perceptual_rgb': forward_return['loss']['perceptual_rgb'],
                                'train/loss/pixel_ir': forward_return['loss']['pixel_ir'],
                                'train/loss/perceptual_ir': forward_return['loss']['perceptual_ir'],
                                'train/loss/det_reg': forward_return['loss']['det_regression'],
                                'train/loss/det_class': forward_return['loss']['det_classification'],
                                'train/loss/det_objectness': forward_return['loss']['det_objectness'],
                                'train/loss/det_rpn_box_reg': forward_return['loss']['det_rpn_box_reg'],
                                'train/loss/det_bbox_ctrness': forward_return['loss']['det_bbox_ctrness'],
                                'train/loss/det_total': forward_return['loss']['det_total'],
                                'train/loss/total': forward_return['loss']['total'], 
                                'train/loss/step': self.train_loss_step,
        })
        self.train_loss_step = self.train_loss_step + 1

        # if((batch_idx % 100) == 1):
        #     self.wandb_logger.log({"train/media/input_ir": [wandb.Image(forward_return['output']['imgs_ir'], caption="train/input_ir")],
        #                             "train/media/input_rgb": [wandb.Image(forward_return['output']['imgs_rgb'], caption="train/input_rgb")],
        #                             "train/media/output_hal": [wandb.Image(forward_return['output']['imgs_hallucinated'], caption="train/output")],
        #                             "train/media/output_hal_det": [wandb.Image(forward_return['output']['det_hal'], caption="train/output_hal_det")],
        #                             "train/media/output_rgb_det": [wandb.Image(forward_return['output']['det_rgb'], caption="train/output_rgb_det")],
        #                             "train/media/output_ir_det": [wandb.Image(forward_return['output']['det_ir'], caption="train/output_ir_det")],
        #                             "train/media/input_ir_samples": [wandb.Image(im) for im in forward_return['output']['imgs_ir']],
        #                             "train/media/input_rgb_samples": [wandb.Image(im) for im in forward_return['output']['imgs_rgb']],
        #                             "train/media/output_hal_samples": [wandb.Image(im) for im in forward_return['output']['imgs_hallucinated']],
        #                             "train/media/output_hal_det_samples": [wandb.Image(im) for im in forward_return['output']['det_hal']],
        #                             "train/media/output_rgb_det_samples": [wandb.Image(im) for im in forward_return['output']['det_rgb']],
        #                             "train/media/output_ir_det_samples": [wandb.Image(im) for im in forward_return['output']['det_ir']],
        #                             "train/media/step" : self.train_media_step,
        #                         })
        #     self.train_media_step = self.train_media_step + 1


        return forward_return['loss']['total']


    def validation_step(self, val_batch, batch_idx):

        imgs_rgb, targets_rgb, imgs_ir, targets_ir = val_batch

        forward_return =  self.forward_step(imgs_rgb, targets_rgb, imgs_ir, targets_ir, batch_idx, step='val')
        
        self.log('val_loss', forward_return['loss']['total'], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.wandb_logger.log({ 'valid/loss/pixel_rgb': forward_return['loss']['pixel_rgb'], 
                        'valid/loss/perceptual_rgb': forward_return['loss']['perceptual_rgb'],
                        'valid/loss/pixel_ir': forward_return['loss']['pixel_ir'],
                        'valid/loss/perceptual_ir': forward_return['loss']['perceptual_ir'],
                        'valid/loss/det_reg': forward_return['loss']['det_regression'],
                        'valid/loss/det_class': forward_return['loss']['det_classification'],
                        'valid/loss/det_objectness': forward_return['loss']['det_objectness'],
                        'valid/loss/det_rpn_box_reg': forward_return['loss']['det_rpn_box_reg'],
                        'valid/loss/det_bbox_ctrness': forward_return['loss']['det_bbox_ctrness'],
                        'valid/loss/det_total': forward_return['loss']['det_total'],
                        'valid/loss/total': forward_return['loss']['total'],
                        'valid/loss/step': self.valid_loss_step,
                    })
        self.valid_loss_step = self.valid_loss_step + 1

        # if((batch_idx % 100) == 1):
        #     self.wandb_logger.log({"valid/media/input_ir": [wandb.Image(forward_return['output']['imgs_ir'], caption="valid/input_ir")],
        #                             "valid/media/input_rgb": [wandb.Image(forward_return['output']['imgs_rgb'], caption="valid/input_rgb")],
        #                             "valid/media/output_hal": [wandb.Image(forward_return['output']['imgs_hallucinated'], caption="valid/output_hal")], 
        #                             "valid/media/output_hal_det": [wandb.Image(forward_return['output']['det_hal'], caption="valid/output_hal_det")],
        #                             "valid/media/output_rgb_det": [wandb.Image(forward_return['output']['det_rgb'], caption="valid/output_rgb_det")],
        #                             "valid/media/output_ir_det": [wandb.Image(forward_return['output']['det_ir'], caption="valid/output_ir_det")],
        #                             "valid/media/input_ir_samples": [wandb.Image(im) for im in forward_return['output']['imgs_ir']],
        #                             "valid/media/input_rgb_samples": [wandb.Image(im) for im in forward_return['output']['imgs_rgb']],
        #                             "valid/media/output_hal_samples": [wandb.Image(im) for im in forward_return['output']['imgs_hallucinated']],
        #                             "valid/media/output_hal_det_samples": [wandb.Image(im) for im in forward_return['output']['det_hal']],
        #                             "valid/media/output_rgb_det_samples": [wandb.Image(im) for im in forward_return['output']['det_rgb']],
        #                             "valid/media/output_ir_det_samples": [wandb.Image(im) for im in forward_return['output']['det_ir']],
        #                             "valid/media/step" : self.valid_media_step,
        #                         })
        #     self.valid_media_step = self.valid_media_step + 1

        return forward_return['loss']['total']


    def on_validation_epoch_end(self):
        
        map_rgb = Utils.filter_dictionary(self.valid_metrics_detection_map_rgb.compute(), {'map_50', 'map_75', 'map'})
        map_hall = Utils.filter_dictionary(self.valid_metrics_detection_map_hall.compute(), {'map_50', 'map_75', 'map'})
        map_ir = Utils.filter_dictionary(self.valid_metrics_detection_map_ir.compute(), {'map_50', 'map_75', 'map'})
    

        self.wandb_logger.log({
                        'valid/metrics/map_rgb': map_rgb,
                        'valid/metrics/map_hall': map_hall,
                        'valid/metrics/map_ir': map_ir,
                        'valid/metrics/step': self.valid_epoch,
                    })

        if(self.best_valid_map_50 < map_hall['map_50'] and self.current_epoch > 0):

            self.best_valid_map_50 = map_hall['map_50']
            self.best_valid_epoch = self.current_epoch

            self.wandb_logger.summary["valid/metrics/map_rgb"] = map_rgb
            self.wandb_logger.summary["valid/metrics/map_hall"] = map_hall
            self.wandb_logger.summary["valid/metrics/map_ir"] = map_ir
            self.wandb_logger.summary["valid/metrics/best_epoch"] = self.best_valid_epoch
            self.wandb_logger.summary["checkpoint_dirpath"] = self.trainer.checkpoint_callback.dirpath

            ckpt_path = os.path.join(
                self.trainer.checkpoint_callback.dirpath, 'best_encoder_decoder_pl.ckpt'
            )
            self.trainer.save_checkpoint(ckpt_path)

        self.log('val_map', map_hall['map'], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        self.valid_metrics_detection_map_rgb.reset()
        self.valid_metrics_detection_map_hall.reset()
        self.valid_metrics_detection_map_ir.reset()

        self.valid_epoch += 1

    def on_train_epoch_end(self):
        self.train_epoch += 1
        

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
        # output_hal_det = torch.Tensor(np.asarray([Utils().plot_each_image(imgs_hallucinated[idx], det, targets_ir[idx], threshold=args.threshold) 
        #                             for idx, det in enumerate(detections_hall)]))

        ## Detector RGB
        _, detections_rgb  = Detector.calculate_loss(self.detector, imgs_rgb, targets_rgb, train_det=False, model_name=self.detector_name)
        # output_rgb_det = torch.Tensor(np.asarray([Utils().plot_each_image(imgs_rgb[idx], det, targets_rgb[idx], threshold=args.threshold) 
        #                             for idx, det in enumerate(detections_rgb)]))

        ## Detector IR
        _, detections_ir = Detector.calculate_loss(self.detector, imgs_ir_three_channel, targets_ir, train_det=False, model_name=self.detector_name)
        # output_ir_det = torch.Tensor(np.asarray([Utils().plot_each_image(imgs_ir_three_channel[idx], det, targets_ir[idx], threshold=args.threshold) 
        #                             for idx, det in enumerate(detections_ir)]))

        self.test_metrics_detection_map_rgb.update(detections_rgb, targets_rgb)
        self.test_metrics_detection_map_hall.update(detections_hall, targets_ir)
        self.test_metrics_detection_map_ir.update(detections_ir, targets_ir)

        # if((batch_idx % 100) == 1):
        #     self.wandb_logger.log({"test/media/input_ir": [wandb.Image(imgs_ir, caption="test/input_ir")],
        #                             "test/media/input_rgb": [wandb.Image(imgs_rgb, caption="test/input_rgb")],
        #                             "test/media/output": [wandb.Image(outs, caption="test/output")],
        #                             "test/media/input_ir_samples": [wandb.Image(im) for im in imgs_ir],
        #                             "test/media/input_rgb_samples": [wandb.Image(im) for im in imgs_rgb],
        #                             "test/media/output_samples": [wandb.Image(im) for im in outs],
        #                         })

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
                                        params=(list(self.encoder_decoder.parameters()) + list(self.detector.parameters())
                                                if self.train_det else self.encoder_decoder.parameters()),
                                                #list(self.detector.parameters()),
                                        lr=self.lr)

        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

        return {
            "optimizer": optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "val_loss",
            }
        }

# Set device
device = Config.cuda_or_cpu() if args.device is None else args.device

# Model
model = EncoderDecoderLit(batch_size=args.batch, 
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
                        )


if(Config.EncoderDecoder.load_encoder_decoder):
    model = EncoderDecoderLit.load_from_checkpoint(checkpoint_path=Config.EncoderDecoder.encoder_decoder_load_path,
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

# saves best model
checkpoint_best_callback = pl.callbacks.ModelCheckpoint(
    save_top_k=1,
    monitor="val_map",
    mode="max",
    dirpath=os.path.join('lightning_logs', args.wandb_project, args.wandb_name, "_".join([args.dataset, args.modality, Config.Detector.name])),
    filename="best",
)

# Save last model
checkpoint_last_callback = pl.callbacks.ModelCheckpoint(
    save_last=True,
    dirpath=os.path.join('lightning_logs', args.wandb_project, args.wandb_name, "_".join([args.dataset, args.modality, Config.Detector.name])),
    filename="last"
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
                            checkpoint_best_callback,
                            checkpoint_last_callback
                    ],
                    deterministic=False,
                    limit_train_batches=args.limit_train_batches,
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

trainer.save_checkpoint(os.path.join(trainer.checkpoint_callback.dirpath,
                                    'encoder_decoder_pl.ckpt'))

torch.save(model.detector.state_dict(),  
            os.path.join(trainer.checkpoint_callback.dirpath, 'detector.bin')
)

torch.save(model.encoder_decoder.state_dict(),  
            os.path.join(trainer.checkpoint_callback.dirpath, 'encoder_decoder.bin')
)

trainer.test(model, dm, ckpt_path="best")

wandb_logger.summary["checkpoint_dirpath"] = trainer.checkpoint_callback.dirpath

wandb_logger.finish()