import torch
import torch.nn.functional as F
import numpy as np
import dask.array as da
import yaml

def kld(p, q):
    """
    Compute the Kullback-Leibler Divergence between two distributions.

    :param p: Probability distribution (histogram) of the reference image.
    :param q: Probability distribution (histogram) of the test image.
    :return: KLD value.
    """
    # Ensure that distributions are non-zero
    eps = 1e-10
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    
    # Normalize distributions
    p = p / p.sum()
    q = q / q.sum()
    
    # Calculate KLD
    kld = torch.sum(p * torch.log(p / q))
    return kld

def normalize_laplacian_level(laplacian_level, kernel_size=3, C=1e-6):
    """
    Normalize a Laplacian level by local standard deviation.

    :param laplacian_level: A single level of the Laplacian pyramid [C,H,W].
    :param kernel_size: Size of the kernel for local statistics.
    :param C: Small constant to avoid division by zero.
    :return: Normalized Laplacian level.
    """
    if len(laplacian_level.shape) == 2:
        laplacian_level = laplacian_level.unsqueeze(0)
    
    padding = kernel_size // 2
    
    # Calculate local mean
    local_mean = F.avg_pool2d(
        laplacian_level.unsqueeze(0),
        kernel_size,
        stride=1,
        padding=padding
    ).squeeze(0)
    
    # Calculate local variance
    squared_diff = (laplacian_level - local_mean) ** 2
    local_var = F.avg_pool2d(
        squared_diff.unsqueeze(0),
        kernel_size,
        stride=1,
        padding=padding
    ).squeeze(0)
    
    # Normalize
    local_std = torch.sqrt(local_var + C)
    normalized = (laplacian_level - local_mean) / local_std
    
    return normalized

def calculate_histogram(image, bins=256, range=(-6, 6)):
    """
    Calculate the histogram of an image.

    :param image: A single level of the Laplacian pyramid (as a torch tensor).
    :param bins: Number of bins in the histogram.
    :param range: The range of values to include in the histogram.
    :return: Normalized histogram representing the probability distribution.
    """
    # Flatten the image
    flattened_image = image.flatten()
    
    # Calculate histogram
    hist = torch.histc(flattened_image, bins=bins, min=range[0], max=range[1])
    
    # Normalize histogram
    hist = hist / hist.sum()
    
    return hist

def gaussian_pyramid(image, max_levels):
    """
    Generate a Gaussian pyramid for an image.
    
    Args:
        image: Input image as a PyTorch tensor [C,H,W]
        max_levels: Maximum number of levels in the pyramid
    Returns:
        List of torch tensors representing the Gaussian pyramid
    """
    # Ensure image has batch dimension [B,C,H,W]
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    pyramid = [image]
    current_image = image
    
    for _ in range(1, max_levels):
        current_image = F.avg_pool2d(current_image, kernel_size=2, stride=2)
        pyramid.append(current_image)
    
    return pyramid

def laplacian_pyramid(gaussian_pyramid):
    """
    Generate a Laplacian pyramid from a Gaussian pyramid.
    
    Args:
        gaussian_pyramid: List of torch tensors [B,C,H,W]
    Returns:
        List of torch tensors representing the Laplacian pyramid
    """
    laplacian_pyramid = []
    
    for i in range(len(gaussian_pyramid) - 1):
        current_size = gaussian_pyramid[i].shape[-2:]
        # Ensure proper dimensions for interpolation
        upsampled = F.interpolate(gaussian_pyramid[i + 1], size=current_size, mode='nearest')
        laplacian = gaussian_pyramid[i] - upsampled
        laplacian_pyramid.append(laplacian)
    
    # Add the last Gaussian level
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

def convert_to_tensor(data):
    """Convert input to proper tensor format [B,C,H,W]"""
    # Convert Dask array to NumPy array
    if isinstance(data, da.Array):
        data = data.compute()
    
    # Convert NumPy array to PyTorch tensor
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    
    # Ensure proper dimensions
    if len(data.shape) == 2:  # [H,W]
        data = data.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif len(data.shape) == 3:  # [C,H,W]
        data = data.unsqueeze(0)  # [1,C,H,W]
    
    return data

def kld_loss(hr_image, sr_image, num_levels=3):
    """
    Calculate KLD loss between HR and SR images.
    
    Args:
        hr_image: High-res image tensor [C,H,W]
        sr_image: Super-res image tensor [C,H,W]
        num_levels: Number of pyramid levels
    Returns:
        KLD value
    """
    # Convert inputs to proper format [B,C,H,W]
    hr_image = convert_to_tensor(hr_image)
    sr_image = convert_to_tensor(sr_image)
    
    # Generate pyramids
    gauss_hr = gaussian_pyramid(hr_image, num_levels)
    gauss_sr = gaussian_pyramid(sr_image, num_levels)
    
    # Generate Laplacian pyramids
    lapl_hr = laplacian_pyramid(gauss_hr)
    lapl_sr = laplacian_pyramid(gauss_sr)
    
    # Calculate KLD for each level
    total_kld = 0
    weights = [0.5, 0.25, 0.25]  # Weights for different pyramid levels
    
    for i, (hr_level, sr_level) in enumerate(zip(lapl_hr, lapl_sr)):
        # Normalize Laplacian levels
        norm_hr = normalize_laplacian_level(hr_level.squeeze(0))
        norm_sr = normalize_laplacian_level(sr_level.squeeze(0))
        
        # Calculate histograms
        hist_hr = calculate_histogram(norm_hr)
        hist_sr = calculate_histogram(norm_sr)
        
        # Calculate KLD and add weighted contribution
        level_kld = kld(hist_hr, hist_sr)
        total_kld += weights[i] * level_kld
    
    return total_kld.item()



