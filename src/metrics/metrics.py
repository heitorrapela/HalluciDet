import torch
import torchmetrics
import torchmetrics.detection # Lazy import
from torchvision.ops import box_iou

# https://github.com/Lightning-AI/metrics/issues/1024 
try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection import MAP
    MeanAveragePrecision = MAP


class Reconstruction():

    class Perceptual():
        
        def __init__(self, metrics=['psnr', 'ssim', 'mssim', 'lpips'], device='cpu'):

            self.psnr = None
            self.ssim = None
            self.mssim = None
            self.lpips = None
            self.device = device

            if('psnr' in metrics):
                self.psnr = self.metric_psnr()
            if('ssim' in metrics):
                self.ssim = self.metric_ssim()
            if('mssim' in metrics):
                self.mssim = self.metric_msssi()
            if('lpips' in metrics):
                self.lpips = self.metric_lpips()


        # PSNR Metric
        def metric_psnr(self):
            return torchmetrics.PeakSignalNoiseRatio().to(self.device)

        # SSIM Metric
        def metric_ssim(self):
            return torchmetrics.StructuralSimilarityIndexMeasure().to(self.device)

            
        # MSSSIM Metric
        def metric_msssi(self):
            return torchmetrics.MultiScaleStructuralSimilarityIndexMeasure().to(self.device)

        # LPIPS Metric
        def metric_lpips(self, net='vgg'):
            return torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type=net).to(self.device)
  

    class Pixel():

        def __init__(self, metrics=['mae', 'mse', 'rmse', 'msle'], device='cpu'):

            self.mae = None
            self.mse = None
            self.rmse = None
            self.msle = None
            self.device = device

            if('mae' in metrics):
                self.mae = self.metric_mae()
            if('mse' in metrics):
                self.mse = self.metric_mse()
            if('rmse' in metrics):
                self.rmse = self.metric_rmse()
            if('msle' in metrics):
                self.msle = self.metric_msle()

        # MAE Metric
        def metric_mae(self): 
            return torchmetrics.MeanAbsoluteError().to(self.device)

        # MSE Metric
        def metric_mse(self):
            return torchmetrics.MeanSquaredError(squared=True).to(self.device)

        # RMSE Metric
        def metric_rmse(self):
            return torchmetrics.MeanSquaredError(squared=False).to(self.device)

        # MSLE Metric
        def metric_msle(self):
            return torchmetrics.MeanSquaredLogError().to(self.device)


class Detection():

    def __init__(self, box_format='xyxy', device='cpu', class_metrics=False):

        self.device = device
        self.map = self.metric_map(box_format=box_format, class_metrics=class_metrics)
        
    
    # IoU Metric
    def iou_bboxes(self, bbox1, bbox2):
        iou_matrix = box_iou(
            torch.Tensor(bbox1)[:, 3:].int(), torch.Tensor(bbox2)[:, 3:].int()
        )
        return iou_matrix.numpy()
        

    # MAP Metric
    def metric_map(self, box_format='xyxy', class_metrics=False):
        return MeanAveragePrecision(class_metrics=class_metrics).to(self.device)