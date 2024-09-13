import torch

class Reconstruction():

    @staticmethod
    def select_loss_perceptual(loss_perceptual='lpips_alexnet'):

        if(loss_perceptual == 'psnr'):
            return Reconstruction.Perceptual.pnsr()
        elif(loss_perceptual == 'ssim'):
            return Reconstruction.Perceptual.ssim()
        elif(loss_perceptual == 'mssim'):
            return Reconstruction.Perceptual.msssim()
        elif(loss_perceptual == 'lpips_alexnet'):
            return Reconstruction.Perceptual.lpips(net='alex')
        elif(loss_perceptual == 'lpips_vgg'):
            return Reconstruction.Perceptual.lpips(net='vgg')
        elif(loss_perceptual == 'lpips_squeeze'):
            return Reconstruction.Perceptual.lpips(net='squeeze')
        
        return None


    class Perceptual():
    
        # PSNR loss
        @staticmethod
        def psnr(max_val=1.0):
            return kornia.losses.PSNRLoss(max_val=max_val)

        # SSIM loss
        @staticmethod
        def ssim(window_size=5):
            return kornia.losses.SSIMLoss(window_size=window_size)

        # MSSSIM loss  
        @staticmethod
        def mssim():
            return kornia.losses.MS_SSIMLoss()

        # LPIPS loss
        @staticmethod
        def lpips(net='alex'):
            return lpips.LPIPS(net=net)
            

    @staticmethod
    def select_loss_pixel(loss_pixel='mse'):

        if(loss_pixel == 'mse'):
            return Reconstruction.Pixel.mse()
        elif(loss_pixel == 'l1'):
            return Reconstruction.Pixel.l1()
        elif(loss_pixel == 'tv'):
            return Reconstruction.Pixel.total_variation()

        return None


    class Pixel():
        
        # MSE loss
        @staticmethod
        def mse():
            return torch.nn.MSELoss()

        # L1 Loss
        @staticmethod
        def l1():
            return torch.nn.L1Loss()

        # Total Variation loss
        @staticmethod
        def total_variation():
            return kornia.losses.TotalVariation()
        