
import argparse
import torch
import os

class Config:

    class Environment:

        N_CORE = "8"
        N_THREADS_TORCH = 8
        N_GPUS = 1
        CUDNN_BENCHMARK = True
        DEBUG = False

    class Optimizer:

        name = 'adam'
        scheduler_step_size = 10
        scheduler_gamma = 0.1
        scheduler_on = True
        gradient_clip_val = 0.5


    class Dataset:
        
        train_valid_split = 0.8
        dataset = 'llvip'

        if dataset == 'kaist': 
            train_path = '../datasets/kaist/train'
            valid_path = None # validation is created on the fly (with same seed you get same imgs)
            test_path = '../datasets/kaist/test'
            ext = '.png'
            n_classes = 2 # person and background
        elif dataset == 'llvip':
            train_path = '../datasets/LLVIP/' # Put the same here, because we need to filter the folders infrared and visible
            valid_path = None
            test_path = '../datasets/LLVIP/'
            ext = '.jpg'
            n_classes = 2
        elif dataset == 'flir':   
            train_path = '../datasets/FLIR_aligned/'
            valid_path = None
            test_path = '../datasets/FLIR_aligned/'
            ext = '.jpg'
            n_classes = 2
        else:
            train_path = None
            valid_path = None
            test_path = None
            ext = '.png'
            n_classes = 2


    class Losses:

        hparams_losses_weights = {
            'pixel_rgb': 0.0,
            'pixel_ir': 0.0,
            'perceptual_rgb': 0.0,
            'perceptual_ir': 0.0,
            'det_regression': 0.1,
            'det_classification': 0.1,
            'det_objectness' : 0.1,
            'det_rpn_box_reg' : 0.1,
            'det_bbox_ctrness' : 0.1,
            'det_masked': 0.0,
        }

        pixel = None
        perceptual = None

    class EncoderDecoder:

        in_channels_encoder = 3
        out_channels_decoder = 3
        decoder_head = 'sigmoid'
        load_encoder_decoder = False # load_encoder_decoder = True
        encoder_decoder_load_path = 'lightning_logs/hallucidet/detector_fasterrcnn_hallucidet_det01reg01_seed123/llvip_ir_fasterrcnn/best.ckpt'


    class Detector:

        train_det = False
        name = 'fasterrcnn'
        pretrained = True
        input_size = 300 # 640 for flir 
        batch_norm_eps = 0.001
        batch_norm_momentum = 0.03
        eval_path = None
        modality = None
        score_threshold = 0.5
    

    @staticmethod
    def argument_parser():
        parser = argparse.ArgumentParser(description='HalluciDet')

        parser.add_argument('--dataset', type=str, default=None, help='llvip/flir')

        parser.add_argument('--train', type=str, default=None, help='Train Dataset Path')

        parser.add_argument('--valid', type=str, default=None, help='Valid Dataset Path')

        parser.add_argument('--test', type=str, default=None, help='Test Dataset Path')

        parser.add_argument('--n-classes', '--n_classes', '--num-classes', '--nclasses', type=int, default=2, help='Number of classes (default: 2)')

        parser.add_argument('--detector', type=str, default='fasterrcnn', help="Model Path (default: fasterrcnn), choices=['fasterrcnn', 'fcos', 'retinanet']")

        parser.add_argument('--pretrained', action='store_true', help='Flag to load pretrained model (default: False)')

        parser.add_argument('--fine-tuning', action='store_true', help='Flag to load finetuning detector (rgb2ir) (default: False)')

        parser.add_argument('--fine-tuning-lp', action='store_true', help='Flag to update only the head, set also ft flag for lower lr (default: False)')
        
        parser.add_argument('--modality', type=str, default='rgb', help='Modality: rgb or ir (default: rgb)')

        parser.add_argument('--threshold', type=float, default=0.5, help='Object detection score threshold (default: 0.5)')

        parser.add_argument('--epochs', type=int, default=10, help='Max number of epochs. (default: 30)')

        parser.add_argument('--lr', type=float, default=None, help='Learning Rate. (default: None, is going to use what is in config files)')

        parser.add_argument('--seed', type=int, default=123, help='Seed value')

        parser.add_argument('--wandb-project', type=str, default="hallucidet", help='Wandb Project Name')

        parser.add_argument('--wandb-name', type=str, default="detector", help='Wandb Run Name')

        parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")

        parser.add_argument("--num-workers", type=int, default=4, help="Number of workers. (default: 4)")

        parser.add_argument("--ext", "--input-ext", type=str, default=None, help='Image Extension (.png, .jpg)')

        parser.add_argument("--output-model", type=str, default="example.ckpt", help='Output model name')

        parser.add_argument("--detector-path", type=str, default=None, help='Pre-trained detector to load (.bin)')

        parser.add_argument("--device", type=str, default=None, help='gpu or cpu')

        parser.add_argument("--fuse-data", type=str, default='none', help='none, addition, attention, cross')

        parser.add_argument("--decoder-backbone", type=str, default='resnet34', help='resnet18, resnet34, resnet50')

        parser.add_argument("--precision", type=int, default=32, help='16 or 32')

        parser.add_argument("--optimizer", type=str, default='adamw', help='adamw, adam, sgd, lion')

        parser.add_argument("--path", type=str, default=None, help='Detector path to fine tuning')

        parser.add_argument("--segmentation-head", type=str, default='sigmoid', help='sigmoid')
         
        parser.add_argument("--pixel", type=str, default=None, help="Pixel Loss (default: None), choices=['mse', 'l1']")

        parser.add_argument("--weight-pixel-rgb", type=float, default=0.0, help="Pixel Loss Weight RGB (default: 0.0)")

        parser.add_argument("--weight-pixel-ir", type=float, default=0.0, help="Pixel Loss Weight IR (default: 0.0)")

        parser.add_argument("--perceptual", type=str, default=None, help="Perceptual Loss (default: None), choices=['psnr', 'ssim', 'msssim', 'lpips_alexnet', 'lpips_vgg', 'lpips_squeeze']")

        parser.add_argument("--weight-perceptual-rgb", type=float, default=0.0, help="Perceptual Loss Weight RGB (default: 0.0)")

        parser.add_argument("--weight-perceptual-ir", type=float, default=0.0, help="Perceptual Loss Weight IR (default: 0.0)")

        parser.add_argument("--weight-det-regression", type=float, default=0.1, help="Weight Detection Regression Loss (default: 0.1)")

        parser.add_argument("--weight-det-classification", type=float, default=0.1, help="Weight Detection Classification Loss (default: 0.1)")

        parser.add_argument("--weight-det-masked", type=float, default=0.0, help="Weight Detection Masked Loss (default: 0.0)")

        parser.add_argument("--weight-det-objectness", type=float, default=0.1, help="Weight Detection Objectness Loss (default: 0.1 detector: fasterrcnn)")

        parser.add_argument("--weight-det-rpn-box-reg", type=float, default=0.1, help="Weight Detection RPN Box Regression Loss (default: 0.1 detector: fasterrcnn)")

        parser.add_argument("--weight-det-bbox-ctrness", type=float, default=0.1, help="Weight Detection Bbox Center-ness Loss (default: 0.1 detector: fcos)")

        parser.add_argument("--image2image-model", type=str, default=None, help="Image2image model path (default: None -> it is going to eval rgb baseline)")

        parser.add_argument('--directly-coco', action='store_true', help='With this flag the pretrain starts without change (default: False)')

        parser.add_argument('--limit-train-batches',  type=float, default=1.0, help='Percentage of training batch size (default: 1.0 - full batch)')

        parser.add_argument('--ablation-flag', action='store_true', help='With this flag we can eval test set every epoch for ablation study (default: False)')
        
        parser.add_argument("--pre-train-path", type=str, default=None, help="HalluciDet Path")

        parser.add_argument("--encoder-depth", type=int, default=5, help='Depth of the encoder 3 to 5. (default: 5)')

        args = parser.parse_args()

        return args


    @staticmethod
    def cuda_or_cpu():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def config_optimizer(optimizer='adamw', params=None, lr=1e-5, momentum=0.9, weight_decay=0.0005):

        if(optimizer == 'sgd'):
            return torch.optim.SGD([
                dict(params=list(params),
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay),
            ])
            
        elif(optimizer == 'adam'):
            return torch.optim.Adam([ 
                dict(params=list(params), 
                    lr=lr),
            ])

        elif(optimizer == 'adamw'):
            return torch.optim.AdamW([ 
                dict(params=list(params), 
                    lr=lr),
            ])

        elif(optimizer == 'lion'):
            try:
                from lion_pytorch import Lion
                
                return Lion([
                    dict(params=list(params), 
                        lr=lr),
                ])
            except:    
                print("Lion optimizer is not installed. Please install it from pip install lion-pytorch")

        elif(optimizer == 'adadelta'):
            return torch.optim.Adadelta([ 
                dict(params=list(params), 
                    lr=lr,
                    ),
            ])
            
        return None


    @staticmethod
    def config_scheduler(optimizer, mode='min', factor=0.1, patience=5):

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)

        lr_scheduler = {
            "scheduler": scheduler, 
            "reduce_on_plateau": True,
            "interval": "epoch",
            "monitor": "val_loss"
        }

        return lr_scheduler

    @staticmethod
    def set_environment():

        os.environ["OMP_NUM_THREADS"] = Config.Environment.N_CORE
        os.environ["OPENBLAS_NUM_THREADS"] = Config.Environment.N_CORE
        os.environ["MKL_NUM_THREADS"] = Config.Environment.N_CORE 
        os.environ["VECLIB_MAXIMUM_THREADS"] = Config.Environment.N_CORE
        os.environ["NUMEXPR_NUM_THREADS"] = Config.Environment.N_CORE
        torch.set_num_threads(Config.Environment.N_THREADS_TORCH)
        torch.backends.cudnn.benchmark = Config.Environment.CUDNN_BENCHMARK
        

    @staticmethod 
    def set_dataset_path(dataset):

        if dataset == 'kaist': 
            train_path = '../datasets/kaist/train'
            valid_path = None 
            test_path = '../datasets/kaist/test'
            ext = '.png'
            n_classes = 2
        elif dataset == 'llvip':
            train_path = '../datasets/LLVIP/'
            valid_path = None 
            test_path = '../datasets/LLVIP/'
            ext = '.jpg'
            n_classes = 2
        elif dataset == 'flir':   
            train_path = '../datasets/FLIR_aligned/'
            valid_path = None 
            test_path = '../datasets/FLIR_aligned/'
            ext = '.jpg'
            n_classes = 2
        else:
            train_path = None
            valid_path = None
            test_path = None
            ext = '.png'
            n_classes = 2

        Config.Dataset.dataset = dataset
        Config.Dataset.train_path = train_path
        Config.Dataset.valid_path = valid_path
        Config.Dataset.test_path = test_path
        Config.Dataset.ext = ext
        Config.Dataset.n_classes = n_classes


    @staticmethod 
    def set_detector(name, train_det=False, pretrained=False, score_threshold=0.5):

        Config.Detector.name = name
        Config.Detector.train_det = train_det
        Config.Detector.pretrained = pretrained
        Config.Detector.score_threshold = score_threshold
        Config.Detector.input_size = 640 if Config.Dataset.dataset == 'flir' else 300
        Config.Losses.label_smoothing = 0.1 if Config.Dataset.dataset == 'flir' else 0.0

    @staticmethod
    def set_loss_weights(args):

        if args.pixel is not None:
            Config.Losses.pixel = args.pixel

        if args.perceptual is not None:
            Config.Losses.perceptual = args.perceptual

        if args.weight_pixel_rgb != 0.0:
            Config.Losses.hparams_losses_weights['pixel_rgb'] = args.weight_pixel_rgb

        if args.weight_pixel_ir != 0.0:
            Config.Losses.hparams_losses_weights['pixel_ir'] = args.weight_pixel_ir

        if args.weight_perceptual_rgb != 0.0:
            Config.Losses.hparams_losses_weights['perceptual_rgb'] = args.weight_perceptual_rgb

        if args.weight_perceptual_ir != 0.0:
            Config.Losses.hparams_losses_weights['perceptual_ir'] = args.weight_perceptual_ir

        if args.weight_det_regression != 0.1:
            Config.Losses.hparams_losses_weights['det_regression'] = args.weight_det_regression

        if args.weight_det_classification != 0.1:
            Config.Losses.hparams_losses_weights['det_classification'] = args.weight_det_classification

        if args.weight_det_masked != 0.0:
            Config.Losses.hparams_losses_weights['det_masked'] = args.weight_det_masked
            
        if args.weight_det_objectness != 0.1:
            Config.Losses.hparams_losses_weights['det_objectness'] = args.weight_det_objectness

        if args.weight_det_rpn_box_reg != 0.1:
            Config.Losses.hparams_losses_weights['det_rpn_box_reg'] = args.weight_det_rpn_box_reg

        if args.weight_det_bbox_ctrness != 0.1:
            Config.Losses.hparams_losses_weights['det_bbox_ctrness'] = args.weight_det_bbox_ctrness