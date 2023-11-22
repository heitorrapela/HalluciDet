import os
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np

# Implementation of CNN-based thermal infrared person detection by domain adaptation
# https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10643/1064308/CNN-based-thermal-infrared-person-detection-by-domain-adaptation/10.1117/12.2304400.full?SSO=1

class CnnBasedThermalInfraredDA(pl.LightningModule):
    def __init__(
        self,
        num_classes=2, 
        model_name='ssd300_vgg16', 
        pretrained=False,
        lr=1e-5,

    ):
        super().__init__()

        self.model = None

        ## ssd300 vgg16 trained -> pretrained on coco
        if(model_name == 'ssd300_vgg16'):
            self.model = torchvision.models.detection.ssd300_vgg16(pretrained=pretrained)

        ## ssdlite320 mobilenet v3 -> pretrained on coco
        elif(model_name == 'ssdlite320_mobilenetv3'):
            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=pretrained)

        ## fasterrcnn_resnet50_fpn
        elif(model_name == 'fasterrcnn_resnet50_fpn'):
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

        ## retinanet_resnet50_fpn
        elif(model_name == 'retinanet_resnet50_fpn'):
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=pretrained)

        else:
            print("Model Name not found (Using ssd300 vgg16 pretrained on coco dataset)")
            self.model = torchvision.models.detection.ssd300_vgg16(pretrained=pretrained)

        self.lr = lr


    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)


    def predict(self, imgs):
        self.eval()
        with torch.no_grad():
            output = self.forward(imgs)

        return output


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


    @staticmethod
    def basic_preprocessing_invert(input, channels=[0, 1, 2]):
        ## Input is a tensor of shape [batch_size, 3, height, width]
        input = input.clone()
        # print(input.shape)
        # for c in channels:
        #     input[:, c, :, :] = torchvision.transforms.functional.invert(input[:, c, :, :])
        input = torchvision.transforms.functional.invert(input)
        
        return input


    @staticmethod
    def basic_preprocessing_blur(input, channels=[0, 1, 2], kernel_size=(3,3), sigma=None):
        ## Input is a tensor of shape [batch_size, 3, height, width]
        input = input.clone()

        # for c in channels:
        #     input[:, c, :, :] = torchvision.transforms.functional.gaussian_blur(input[:, c, :, :], kernel_size=kernel_size, sigma=sigma)
        input = torchvision.transforms.functional.gaussian_blur(input, kernel_size=kernel_size, sigma=sigma)
        
        return input


    @staticmethod
    def basic_preprocessing_histogram_stretching_default(input, channels=[0, 1, 2]):
        ## Input is a tensor of shape [batch_size, 3, height, width]
        input = input.clone()

        for c in channels:
            input[c, :, :] = (input[c, :, :] - input[c, :, :].min()) / (input[c, :, :].max() - input[:, c, :, :].min())
        
        return input


    @staticmethod
    def basic_preprocessing_histogram_stretching(input, channels=[0, 1, 2], beta=0.003):
        ## Input is a tensor of shape [3, height, width]
        # beta = 0.003 cnn-based paper
        input = input.clone()

        for c in channels:
 
            q_min = torch.quantile(input[c, :, :], q=beta)
            q_max = torch.quantile(input[c, :, :], q=1-beta)
            
            input[c, :, :] = (input[c, :, :] - q_min) / (q_max - q_min)
            input[c, :, :] = torch.clamp(input[c, :, :], q_min, q_max)

        return input


    @staticmethod
    def basic_preprocessing_histogram_equalization(input, channels=[0, 1, 2]):
        ## Input is a tensor of shape [batch_size, 3, height, width]
        ## Would be good to plot the dist of each channel
        input = input.clone()
        #for c in channels:
        input = torchvision.transforms.functional.equalize((input*255).type(torch.uint8))
        
        input = input.type(torch.float32) / 255.0

        return input

    @staticmethod
    def basic_preprocessing_invert_stretching(input, channels=[0, 1, 2]):
        ## Consecutive Preprocessing
        ## Input is a tensor of shape [batch_size, 3, height, width]
        input = input.clone()

        # Apply Invert
        input = CnnBasedThermalInfraredDA.basic_preprocessing_invert(input, channels=channels)
        
        # Apply Histogram Stretching
        input = CnnBasedThermalInfraredDA.basic_preprocessing_histogram_stretching(input, channels=channels)
        
        return input


    @staticmethod
    def basic_preprocessing_invert_stretching_blur(input, channels=[0, 1, 2]):
        ## Consecutive Preprocessing
        ## Input is a tensor of shape [batch_size, 3, height, width]
        input = input.clone()

        # Apply Invert + Histogram Stretching
        input = CnnBasedThermalInfraredDA.basic_preprocessing_invert_stretching(input, channels=channels)
        
        # Apply Histogram Blur
        input = CnnBasedThermalInfraredDA.basic_preprocessing_blur(input, channels=channels)

        return input


    @staticmethod
    def basic_preprocessing_invert_equalization(input, channels=[0, 1, 2]):
        ## Consecutive Preprocessing
        ## Input is a tensor of shape [batch_size, 3, height, width]
        input = input.clone()

        # Apply Invert
        input = CnnBasedThermalInfraredDA.basic_preprocessing_invert(input, channels=channels)
        
        # Apply Histogram Equalization
        input = CnnBasedThermalInfraredDA.basic_preprocessing_histogram_equalization(input, channels=channels)
        
        return input        


    @staticmethod
    def basic_preprocessing_invert_equalization_blur(input, channels=[0, 1, 2]):
        ## Consecutive Preprocessing
        ## Input is a tensor of shape [batch_size, 3, height, width]
        input = input.clone()

        # Apply Invert + Histogram Equalization
        input = CnnBasedThermalInfraredDA.basic_preprocessing_invert_equalization(input, channels=channels)
        
        # Apply Blur
        input = CnnBasedThermalInfraredDA.basic_preprocessing_blur(input, channels=channels)

        return input    


    @staticmethod
    def basic_preprocessing_collor_jitter(input, brightness, contrast, saturation, hue, channels=[0, 1, 2]):
        ## Consecutive Preprocessing
        ## Input is a tensor of shape [batch_size, 3, height, width]
        input = input.clone()

        jitter = torchvision.transforms.ColorJitter(brightness=brightness, 
                                                    contrast=contrast, 
                                                    saturation=saturation, 
                                                    hue=hue)

        for c in channels:
            input[c, :, :] = jitter(input[c, :, :])

        return input        


    @staticmethod
    def paralel_combination(input, channel_op=['equalization', 'invert', 'none']):#['none', 'invert', 'equalization']):
        ## Consecutive Preprocessing
        ## Input is a tensor of shape [batch_size, 3, height, width]
        input = input.clone()

        for idx, op in enumerate(channel_op):

            if op == 'none':
                continue
            elif op == 'invert':
                input = CnnBasedThermalInfraredDA.basic_preprocessing_invert(input, channels=[idx])
            elif op == 'equalization':
                input = CnnBasedThermalInfraredDA.basic_preprocessing_histogram_equalization(input, channels=[idx])
            else:
                pass

        # Apply Invert + Histogram Equalization
        # input = CnnBasedThermalInfraredDA.basic_preprocessing_invert_equalization(input, channels=channels)
        
        # Apply Blur
        # input = CnnBasedThermalInfraredDA.basic_preprocessing_blur(input, channels=channels)

        return input

        