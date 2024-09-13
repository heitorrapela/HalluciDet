import torchvision
import torch
import sys 
sys.path.append("./src/custom_segmentation_models/")
import src.custom_segmentation_models.segmentation_models_pytorch as smp

# EncoderDecoder model that wraps segmentation models
class EncoderDecoder():
    def __init__(
        self,
        name='resnet34', 
        encoder_depth=5,
        encoder_weights=None,
        decoder_attention_type=None,
        in_channels=3,
        output_channels=3,
        segmentation_head='sigmoid',
        dropout = 0.2,
        avg2d_flag = True
        ):
        
        self.encoder_decoder = smp.Unet(name,
                    encoder_depth=encoder_depth,
                    encoder_weights=encoder_weights,
                    decoder_attention_type=decoder_attention_type,
                    in_channels=in_channels,
                    classes=output_channels)
        
        if(segmentation_head == 'sigmoid'):
            self.encoder_decoder.segmentation_head[-1] = torch.nn.Sigmoid()

        elif(segmentation_head == 'relu_bn'):
            ## Need to test other variations of this head (batchnorm before and sigmoid)
            self.encoder_decoder.segmentation_head[-1] = torch.nn.Sequential(
                                                    torch.nn.ReLU(),
                                                    torch.nn.BatchNorm2d(self.output_channels),
                                                )
            
        elif(segmentation_head == 'avg_dropout_sigmoid'):  

            self.encoder_decoder.segmentation_head[-1] = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1) if avg2d_flag else torch.nn.Identity(),
                torch.nn.Dropout(p=dropout, inplace=True),    
                torch.nn.Sigmoid()
            )  

    @staticmethod
    def normalization(imgs):
        # imagenet normalization
        for i in range(len(imgs)):
            imgs[i] = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                            std=(0.229, 0.224, 0.225))(imgs[i])
        return imgs