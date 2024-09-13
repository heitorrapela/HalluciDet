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