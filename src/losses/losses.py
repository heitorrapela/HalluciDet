import torch

class Reconstruction():

    @staticmethod
    def select_loss_perceptual(loss_perceptual='lpips_alexnet'):

        if(loss_perceptual == 'lpips_alexnet'):
            return Reconstruction.Perceptual.lpips(net='alex')
        elif(loss_perceptual == 'lpips_vgg'):
            return Reconstruction.Perceptual.lpips(net='vgg')
        elif(loss_perceptual == 'lpips_squeeze'):
            return Reconstruction.Perceptual.lpips(net='squeeze')
        
        return None


    class Perceptual():
    
        # LPIPS loss
        @staticmethod
        def lpips(net='alex'):
            import lpips
            return lpips.LPIPS(net=net)
            

    @staticmethod
    def select_loss_pixel(loss_pixel='mse'):

        if(loss_pixel == 'mse'):
            return Reconstruction.Pixel.mse()
        elif(loss_pixel == 'l1'):
            return Reconstruction.Pixel.l1()

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
        
        