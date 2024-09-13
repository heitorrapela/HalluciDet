import torchvision
import torch
from functools import partial
import sys
import math
# Implementation of forward pass for the detectors to calculate loss and use then in the training of our model
from src.utils.eval_forward_fasterrcnn import eval_forward_fasterrcnn
from src.utils.eval_forward_retinanet import eval_forward_retinanet
from src.utils.eval_forward_fcos import eval_forward_fcos


sys.path.append("./src/models/")
from src.models.custom_generalized_transform import CustomGeneralizedRCNNTransform

def _xavier_init(conv: torch.nn.Module):
    for layer in conv.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


# Detector model from torchsivion
class Detector():
    def __init__(
        self,
        name='fasterrcnn_resnet50_fpn', 
        pretrained=True,
        n_classes=2, 
        size=300,
        batch_norm_eps=0.001,
        batch_norm_momentum=0.03,
        eval_path=None,
        modality=None,
        directly_coco=False,
        ):
        
        # If pretrained is True, then load the pretrained model on coco dataset
        self.detector = Detector.select_detector(detector_name=name, pretrained=pretrained)
        
        if not directly_coco:
            
            self.detector.transform = self.change_generalized_transform(min_size=size,
                                                                        max_size=size,  
                                                                        image_mean=[0.0],
                                                                        image_std=[1.0],
                                                                        size_divisible=1,
                                                                        fixed_size=(size, size)) 


            if('fasterrcnn' in name):

                in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
                self.detector.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)
                _xavier_init(self.detector.roi_heads)

            elif('fcos' in name or 'retinanet' in name):

                out_channels = self.detector.head.classification_head.conv[0].out_channels
                num_anchors = self.detector.head.classification_head.num_anchors
                self.detector.head.classification_head.num_classes = n_classes

                cls_logits = torch.nn.Conv2d(out_channels, num_anchors * n_classes, kernel_size=3, stride=1, padding=1)
                torch.nn.init.normal_(cls_logits.weight, std=0.01)
                torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))
                self.detector.head.classification_head.cls_logits = cls_logits


            if eval_path is not None and '.bin' in eval_path:
                self.detector.load_state_dict(torch.load(eval_path))
            else:
                if(eval_path is not None and '.ckpt' in eval_path):
                    eval_path = (eval_path.split('.ckpt')[0] + '.bin').replace('best', 'detector')
                    try:
                        self.detector.load_state_dict(torch.load(eval_path))
                    except:
                        # We opted to make the pytorch lightning load, thats why we ignore this part
                        # The .bin model save is a good one, but the pytorch lightning save is better
                        print("Select model is not compatible with the detector (Requires: .bin dict)")


    # Calculate the parameters for the classification head
    def calculate_parameters_for_head(self, size=300, batch_norm_eps=0.001, batch_norm_momentum=0.03):

        in_channels = torchvision.models.detection._utils.retrieve_out_channels(self.detector.backbone, (size, size))
        num_anchors = self.detector.anchor_generator.num_anchors_per_location()
        norm_layer  = partial(torch.nn.BatchNorm2d, eps=batch_norm_eps, momentum=batch_norm_momentum)

        return in_channels, num_anchors, norm_layer



    ## The default parameters are for fasterrcnn_resnet50_fpn vgg16 but the mean and std is defined for mean 0 and std 1
    def change_generalized_transform(self, min_size=300, max_size=300, image_mean=[0.0], image_std=[1.0], size_divisible=1, fixed_size=(300, 300)):
        return CustomGeneralizedRCNNTransform(min_size=min_size, 
                                            max_size=max_size, 
                                            image_mean=image_mean, 
                                            image_std=image_std,
                                            size_divisible=size_divisible,
                                            fixed_size=fixed_size
                                            )


    @staticmethod
    def calculate_loss(detector, outs, targets, train_det=False, model_name='fasterrcnn'):

        if('fasterrcnn' in model_name):
            losses_det, detections = eval_forward_fasterrcnn(detector, outs, targets, train_det=train_det, model_name=model_name)

        elif('retinanet' in model_name):
            losses_det, detections = eval_forward_retinanet(detector, outs, targets, train_det=train_det, model_name=model_name)

        elif('fcos' in model_name):
            losses_det, detections = eval_forward_fcos(detector, outs, targets, train_det=train_det, model_name=model_name)



        return losses_det, detections


    # Select the detector model
    @staticmethod
    def select_detector(detector_name='fasterrcnn_resnet50_fpn', pretrained=True):
        
        ## fasterrcnn_resnet50_fpn
        ## retinanet_resnet50_fpn
        ## fcos_resnet50_fpn
        
        if(detector_name == 'fasterrcnn' or detector_name == 'fasterrcnn_resnet50_fpn'):
            return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

        elif(detector_name == 'retinanet' or detector_name == 'retinanet_resnet50_fpn'):
            return torchvision.models.detection.retinanet_resnet50_fpn(pretrained=pretrained)
        
        elif(detector_name == 'fcos' or detector_name == 'fcos_resnet50_fpn'):
            return torchvision.models.detection.fcos_resnet50_fpn(pretrained=pretrained)
        
        else:
            print("Model Name not found (Using fasterrcnn_resnet50_fpn")

        return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)