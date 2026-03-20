from PIL import Image
import numpy as np
import torch
import lpips
from faceutils.ssim import ssim

def calculate_ssim(x: torch.Tensor, y: torch.Tensor):
    """
    X, Y: in shape (N, C, H, W), the pixel value range should be [0.0, 1.0]
    output: the ssim of each image in shape (N). ranges from 0 to 1, where 1 means X and Y are identical
    """
    if x.shape != y.shape:
        raise ValueError("Images must have the same dimensions.")
    score = ssim(x, y, data_range=1.0, size_average=False)
    return score

def calculate_lpips(x: torch.Tensor, y: torch.Tensor, net='vgg'):
    """
    X, Y: in shape (N, C, H, W), the pixel value range should be [-1.0, 1.0]
    output: the ssim of each image in shape (N). output >= 0, where 0 means identical
    """
    if x.shape != y.shape:
        raise ValueError("Images must have the same dimensions.")
    loss_fn = lpips.LPIPS(net=net, verbose=False).to(x.device)  # 'alex' is faster, 'vgg' is more accurate
    lpips_score = loss_fn(x, y).squeeze()
    return lpips_score

if __name__ == "__main__":
    img1_path = "/tmp2/r10922106/diff_attack/src/logs/03-13_03:50/161->603/src161_4914.jpg"
    img2_path = "/tmp2/r10922106/diff_attack/src/logs/03-13_03:50/161->603/adv161_4914.jpg"

    res = 256

    img1 = Image.open(img1_path).convert('RGB').resize((res, res))
    img2 = Image.open(img2_path).convert('RGB').resize((res, res))

    img1 = np.array(img1) / 127.5 - 1
    img2 = np.array(img2) / 127.5 - 1

    img1 = img1.transpose((2, 0, 1))
    img2 = img2.transpose((2, 0, 1))

    print(img1.shape, img2.shape)
    print(type(img1))
    
    ssim_score = calculate_ssim(img1, img2)
    lpips_score = calculate_lpips(img1_path, img2_path)
    
    print(f"SSIM Score: {ssim_score:.4f}")
    print(f"LPIPS Score: {lpips_score:.4f}")
