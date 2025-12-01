import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

class SMD2Calculator:
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def img_trans(self, image_path):
        """Image transformation to tensor suitable for SMD2 calculation"""
        image = Image.open(image_path).convert('L')  # convert image to grayscale
        return self.transform(image).unsqueeze(0)  # add batch dimension

    def calculate_smd2(self, image_tensor):
        """
        计算图像的 SMD2 值
        :param image_tensor: PyTorch tensor, shape: (1, 1, H, W)
        :return: SMD2 value
        """
        # 计算水平和垂直梯度
        grad_x = F.conv2d(image_tensor, self.sobel_x, padding=1)
        grad_y = F.conv2d(image_tensor, self.sobel_y, padding=1)

        # 计算梯度平方和
        grad_squared_sum = grad_x ** 2 + grad_y ** 2

        # 计算 SMD2 值
        smd2_value = grad_squared_sum.mean().item()

        return smd2_value

    def calculate_smd2_from_path(self, image_path):
        """
        计算给定图像路径的 SMD2 值
        :param image_path: Path to the image file
        :return: SMD2 value
        """
        image_tensor = self.img_trans(image_path).to(self.device)
        return self.calculate_smd2(image_tensor)



if __name__ == "__main__":
    image_path = "/data/zdr/SR/Real-ESRGAN/zdr/test_results/esrgan_test/x4_DF2K+OST_batch6/net_g_50000/00001_out.png"
    
    smd2_calculator = SMD2Calculator()
    smd2_value = smd2_calculator.calculate_smd2_from_path(image_path)
    print(f"SMD2 Value: {smd2_value}")
