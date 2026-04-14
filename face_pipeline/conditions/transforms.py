from PIL import Image, ImageFilter
import numpy as np


def apply_downsample(image_rgb, downsample_scale, blur_radius):
    width, height = image_rgb.size
    resized_width = max(1, int(width * downsample_scale))
    resized_height = max(1, int(height * downsample_scale))
    degraded_image = image_rgb.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    degraded_image = degraded_image.resize((width, height), resample=Image.Resampling.BILINEAR)
    if blur_radius > 0:
        degraded_image = degraded_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return degraded_image


def apply_gaussian_noise(image_rgb, gaussian_sigma):
    image_array = np.asarray(image_rgb).astype(np.float32)
    image_array += np.random.normal(0, gaussian_sigma, image_array.shape).astype(np.float32)
    noisy_image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image_array)

