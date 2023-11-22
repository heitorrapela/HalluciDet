import torch
import torchmetrics
import torchmetrics.detection # Lazy import
import kornia
from kornia.metrics.average_meter import AverageMeter
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
        return MeanAveragePrecision(class_metrics=class_metrics).to(self.device)#(box_format=box_format).to(self.device)


# https://github.com/Cartucho/mAP (lamr, fppi)
# https://github.com/mrkieumy/task-conditioned/blob/master/plot_LAMR.py
# https://github.com/thollamadugulavanya/project/blob/29b7db62d16e30acacf3267b214e95e98b289a27/tools/ECPB/statistics.py
# https://github.com/Lyken17/pytorch-OpCounter 


if __name__ == '__main__':

    from PIL import Image
    from torchvision import transforms

    from pytorch_lightning import seed_everything
    seed_everything(42)

    import torch

    ## load two images
    torch.manual_seed(42)
    input = Image.open('../input.png').convert('RGB').resize((512,640))
    torch.manual_seed(42)
    #output = Image.open('../output.png').convert('RGB')

    input = (transforms.ToTensor()(input)).unsqueeze(0)#(input/.255).unsqueeze(0)#(transforms.ToTensor()(input)/255.).unsqueeze(0)
    output = input.clone()*0.75
    #output = (transforms.ToTensor()(output)/255.).unsqueeze(0)

    

    # torch.manual_seed(42)
    # input = torch.rand((1, 3, 640, 512))
    # torch.manual_seed(42)
    # output = torch.rand((1, 3, 640, 512))*0.75
    
    print(input.shape)
    print(output.shape)


    # pixel_metrics = Reconstruction.Perceptual(metrics=['psnr', 'ssim', 'mssim', 'lpips'])
    # pixel_metrics = Reconstruction.Perceptual(metrics=['psnr', 'ssim', 'mssim'])
    pixel_metrics = Reconstruction.Perceptual(metrics=['ssim'])
    
    final_v = 0
    j = 0

    debug_torchmetrics_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

    for i in range(3): 
        j = j+1
        output2 = torch.clip(output+0.02*i, 0, 1)
        
        # pixel_metrics.psnr.update(input, output2)
        # pixel_metrics.ssim.reset()
        # pixel_metrics.ssim_test = AverageMeter()
        pixel_metrics.ssim.update(input, output2)
        debug_torchmetrics_ssim(input, output2) #torch.clip(output+0.02*i, 0, 1))

        calc = pixel_metrics.function_metric_ssim(input, output2) #torch.clip(output+0.02*i, 0, 1))
        final_v = final_v + calc
        pixel_metrics.ssim_test.update(calc, n=1)

        # pixel_metrics.mssim.update(input, torch.clip(output+0.02*i, 0, 1))
        # pixel_metrics.lpips.update(input, torch.clip(output+0.02*i, 0, 1))
        print("BATCH ", i)
        # print(pixel_metrics.psnr.compute())
        print("Wrapper Torchmetrics Classe: ", pixel_metrics.ssim.compute())
        print("Torchmetrics Classe", debug_torchmetrics_ssim.compute())
        print("Torchmetrics funcional cm average meter: ", pixel_metrics.ssim_test.avg)
        print("Manual run average ", final_v/j)
        # print(pixel_metrics.mssim.compute())
        # print(pixel_metrics.lpips.compute())
        # print()

    print("MEAN OF THE BATCH:")
    # print(pixel_metrics.psnr.compute())
    print("Wrapper Torchmetrics Classe: ", pixel_metrics.ssim.compute())
    print("Torchmetrics Classe", debug_torchmetrics_ssim.compute())
    print("Torchmetrics funcional cm average meter: ", pixel_metrics.ssim_test.avg)
    print("Manual run average ", final_v/3)
    # print(pixel_metrics.mssim.compute())
    # print(pixel_metrics.lpips.compute())

    # pixel_metrics.psnr.reset()
    # pixel_metrics.ssim.reset()
    # pixel_metrics.mssim.reset()
    # pixel_metrics.lpips.reset()

    # print("epoch 2")
    # import time
    # time.sleep(1)
    # for i in range(20): 

    #     pixel_metrics.psnr.update(input, torch.clip(output+0.02*i, 0, 1))
    #     pixel_metrics.ssim.update(input, torch.clip(output+0.02*i, 0, 1))
    #     pixel_metrics.mssim.update(input, torch.clip(output+0.02*i, 0, 1))
    #     pixel_metrics.lpips.update(input, torch.clip(output+0.02*i, 0, 1))
    #     print("BATCH ", i)
    #     print(pixel_metrics.psnr.compute())
    #     print(pixel_metrics.ssim.compute())
    #     print(pixel_metrics.mssim.compute())
    #     print(pixel_metrics.lpips.compute())
    #     print()

    # print("MEAN OF THE BATCH:")
    # print(pixel_metrics.psnr.compute())
    # print(pixel_metrics.ssim.compute())
    # print(pixel_metrics.mssim.compute())
    # print(pixel_metrics.lpips.compute())

    # pixel_metrics.psnr.reset()
    # pixel_metrics.ssim.reset()
    # pixel_metrics.mssim.reset()
    # pixel_metrics.lpips.reset()


    # pixel_metrics.psnr.update(input, output+0.02*3)
    # pixel_metrics.ssim.update(input, output+0.02*3)

    # print(pixel_metrics.psnr.compute())
    # print(pixel_metrics.ssim.compute())

    # pixel_metrics = Reconstruction.Pixel(metrics=['mae', 'mse', 'rmse', 'msle'])
    
    # for i in range(2): 

    #     pixel_metrics.mae.update(input, output+0.02*i)
    #     pixel_metrics.mse.update(input, output+0.02*i)
    #     pixel_metrics.rmse.update(input, output+0.02*i)
    #     pixel_metrics.msle.update(input, output+0.02*i)
    #     print("BATCH ", i)
    #     print(pixel_metrics.mae.compute())
    #     print(pixel_metrics.mse.compute())
    #     print(pixel_metrics.rmse.compute())
    #     print(pixel_metrics.msle.compute())
    #     print()

    # print("MEAN OF THE BATCH:")
    # print(pixel_metrics.mae.compute())
    # print(pixel_metrics.mse.compute())
    # print(pixel_metrics.rmse.compute())
    # print(pixel_metrics.msle.compute())

    # print(Reconstruction.Perceptual.ssim(input.unsqueeze(0), output.unsqueeze(0)))
    # print(Reconstruction.Perceptual.ssim(output.unsqueeze(0), input.unsqueeze(0)))

    # print(Reconstruction.Perceptual.psnr(input.unsqueeze(0), output.unsqueeze(0)))
    # print(Reconstruction.Perceptual.psnr(output.unsqueeze(0), input.unsqueeze(0)))

    # print(Reconstruction.Perceptual.msssi(input.unsqueeze(0), output.unsqueeze(0)))
    # print(Reconstruction.Perceptual.msssi(output.unsqueeze(0), input.unsqueeze(0)))

    # print(Reconstruction.Perceptual.lpips(input.unsqueeze(0), output.unsqueeze(0)))
    # print(Reconstruction.Perceptual.lpips(output.unsqueeze(0), input.unsqueeze(0)))

    # print(Reconstruction.Perceptual.lpips(input.unsqueeze(0), output.unsqueeze(0), net='alex'))
    # print(Reconstruction.Perceptual.lpips(output.unsqueeze(0), input.unsqueeze(0), net='alex'))

    # print(Reconstruction.Perceptual.lpips(input.unsqueeze(0), output.unsqueeze(0), net='squeeze'))
    # print(Reconstruction.Perceptual.lpips(output.unsqueeze(0), input.unsqueeze(0), net='squeeze'))

    # print(Reconstruction.Pixel.mae(input.unsqueeze(0), output.unsqueeze(0)))
    # print(Reconstruction.Pixel.mae(output.unsqueeze(0), input.unsqueeze(0)))
    
    # print(Reconstruction.Pixel.mse(input.unsqueeze(0), output.unsqueeze(0)))
    # print(Reconstruction.Pixel.mse(output.unsqueeze(0), input.unsqueeze(0)))
    
    # print(Reconstruction.Pixel.rmse(input.unsqueeze(0), output.unsqueeze(0)))
    # print(Reconstruction.Pixel.rmse(output.unsqueeze(0), input.unsqueeze(0)))

    # print(Reconstruction.Pixel.msle(input.unsqueeze(0), output.unsqueeze(0)))
    # print(Reconstruction.Pixel.msle(output.unsqueeze(0), input.unsqueeze(0)))
    