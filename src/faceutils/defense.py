import torch
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
import torchvision

def median_blur(x: torch.Tensor, kernel_size: int=7) -> torch.Tensor:
    """
    Applies median blur to a batch of images (B, C, H, W) using a square kernel.

    x: Input tensor with value range [-1, 1] and shape (B, C, H, W).
    kernel_size: Size of the filter kernel (must be odd).
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    
    B, C, H, W = x.shape
    pad = kernel_size // 2

    # Pad the input tensor
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')

    # Unfold to get sliding local blocks (B, C, kH*kW, H_out * W_out)
    x_unf = x.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    x_unf = x_unf.contiguous().view(B, C, H, W, -1)

    # Take median along the last dimension (the kernel window)
    median = x_unf.median(dim=-1)[0] # median returns (values, indices), we take values only
    return median


def feature_squeezing(x: torch.Tensor, bit_depth: int=3):
    """
    Applies feature squeezing / bit-depth reduction to the input image batch x.

    x: Input tensor with value range [-1, 1] and shape (B, C, H, W).
    bit_depth: Number of bits to keep, starting from the msb.
    """
    shift_bits = 8 - bit_depth
    x = ((x / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8) # to [0, 255]
    x = x.bitwise_right_shift(shift_bits)
    x = x.bitwise_left_shift(shift_bits)
    x = x.to(torch.float32) / 127.5 - 1 # back to [-1, 1]
    return x
    

def gaussian_blur(x: torch.Tensor, kernel_size: int=7):
    """
    Applies Gaussian blur to the input image batch x.

    x: Input tensor with value range [-1, 1] and shape (B, C, H, W).
    kernel_size: Size of the filter kernel (must be odd).
    """
    return ttf.gaussian_blur(x, kernel_size)


def jpeg_defense(x: torch.Tensor, quality: int=5):
    """
    Applies JPEG defense to the input image batch x.

    x: Input tensor with value range [-1, 1] and shape (B, C, H, W).
    quality: The JPEG compression quality. Should be and integer in [0, 100].
    """
    device = x.device
    x = x.cpu()
    B, C, H, W = x.shape
    jpegs = []
    for i in range(B):
        x_i = x[i, :, :, :]
        x_i = ((x_i / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        x_i_jpeg_bytes = torchvision.io.encode_jpeg(x_i, quality=quality)
        x_i_jpeg = torchvision.io.decode_jpeg(x_i_jpeg_bytes)
        x_i_jpeg = x_i_jpeg.to(torch.float32) / 127.5 - 1
        jpegs.append(x_i_jpeg)

    return torch.stack(jpegs).to(device)


if __name__ == "__main__":
    from PIL import Image

    img = Image.open("../../datasets/celebahq_all_images/test/047073.jpg")
    img = ttf.to_tensor(img).unsqueeze(0) * 2 - 1
    
    img_defense = (feature_squeezing(img).squeeze() + 1) * 127.5
    img_defense = img_defense.permute(1, 2, 0).to(torch.uint8).numpy()
    img_defense = Image.fromarray(img_defense)
    img_defense.save("img_defense.png")