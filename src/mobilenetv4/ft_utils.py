from __future__ import annotations

import cv2
import numpy as np
import torch


def generate_ft_from_tensor(
    image_tensor: torch.Tensor, fourier_size: tuple[int, int] = (28, 28)
) -> torch.Tensor:
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)

    if img_np.shape[2] == 3:
        img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img_gray = (img_np[:, :, 0] * 255).astype(np.uint8)

    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift) + 1)

    maxx = np.max(fimg)
    minn = np.min(fimg)
    fimg = (fimg - minn + 1) / (maxx - minn + 1)

    fimg_resized = cv2.resize(fimg, fourier_size)
    fourier_tensor = torch.from_numpy(fimg_resized).float()
    fourier_tensor = torch.unsqueeze(fourier_tensor, 0)

    return fourier_tensor


def generate_ft_batch(
    images: torch.Tensor, fourier_size: tuple[int, int] = (28, 28)
) -> torch.Tensor:
    batch_size = images.shape[0]
    fourier_maps = []

    for i in range(batch_size):
        fourier_map = generate_ft_from_tensor(images[i], fourier_size)
        fourier_maps.append(fourier_map)

    return torch.stack(fourier_maps, dim=0)
