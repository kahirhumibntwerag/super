import torch
import torch.nn.functional as F
import numpy as np
import dask.array as da

def kld(p, q):
    """
    Compute the Kullback-Leibler Divergence between two distributions.

    :param p: Probability distribution (histogram) of the reference image.
    :param q: Probability distribution (histogram) of the test image.
    :return: KLD value.
    """
    # Ensure that q is nonzero to avoid division by zero
    q = q.clamp(min=1e-10)
    p = p.clamp(min=1e-10)

    kld = torch.sum(p * torch.log(p / q))
    return kld

def normalize_laplacian_level(laplacian_level, kernel_size=3, C=1e-5):
    """
    Normalize a Laplacian level by local standard deviation.

    :param laplacian_level: A single level of the Laplacian pyramid (as a torch tensor).
    :param kernel_size: Size of the kernel for local mean and standard deviation.
    :param C: A small constant to avoid division by zero.
    :return: Normalized Laplacian level.
    """
    if len(laplacian_level.shape) == 2:  # Single channel image
        laplacian_level = laplacian_level.unsqueeze(0).unsqueeze(0)
    elif len(laplacian_level.shape) == 3:  # Add batch dimension
        laplacian_level = laplacian_level.unsqueeze(0)

    padding = kernel_size // 2

    # Calculate local mean
    local_mean = F.avg_pool2d(laplacian_level, kernel_size, stride=1, padding=padding)

    # Calculate local variance
    local_var = F.avg_pool2d(torch.pow(laplacian_level, 2), kernel_size, stride=1, padding=padding) - torch.pow(local_mean, 2)

    # Ensure variance is non-negative
    local_var = torch.relu(local_var)

    # Calculate local standard deviation
    local_std = torch.sqrt(local_var + C)

    # Normalize the Laplacian level
    normalized_laplacian = (laplacian_level - local_mean) / local_std

    return normalized_laplacian.squeeze()

def calculate_histogram(image, bins=256, range=(-1, 1)):
    """
    Calculate the histogram of an image.

    :param image: A single level of the Laplacian pyramid (as a torch tensor).
    :param bins: Number of bins in the histogram.
    :param range: The range of values to include in the histogram.
    :return: Normalized histogram representing the probability distribution of pixel intensities.
    """
    # Flatten the image and compute the histogram
    flattened_image = image.flatten()

    # Convert min and max to numbers (scalars)
    hist_min = range[0] if isinstance(range[0], (int, float)) else range[0].item()
    hist_max = range[1] if isinstance(range[1], (int, float)) else range[1].item()

    hist = torch.histc(flattened_image, bins=bins, min=hist_min, max=hist_max)

    # Normalize the histogram to get a probability distribution
    hist = hist / torch.sum(hist)

    return hist

def gaussian_pyramid(image, max_levels):
    """
    Generate a Gaussian pyramid for an image.

    :param image: Input image as a PyTorch tensor.
    :param max_levels: Maximum number of levels in the pyramid.
    :return: List of torch tensors representing the Gaussian pyramid.
    """
    pyramid = [image]
    current_image = image
    for _ in range(1, max_levels):
        current_image = F.avg_pool2d(current_image, kernel_size=2, stride=2, padding=0, ceil_mode=False)
        pyramid.append(current_image)
    return pyramid

def laplacian_pyramid(gaussian_pyramid):
    """
    Generate a Laplacian pyramid from a Gaussian pyramid.

    :param gaussian_pyramid: List of torch tensors representing the Gaussian pyramid.
    :return: List of torch tensors representing the Laplacian pyramid.
    """
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        current_size = gaussian_pyramid[i].shape[-2:]
        upsampled = F.interpolate(gaussian_pyramid[i + 1], size=current_size, mode='nearest')
        laplacian = gaussian_pyramid[i] - upsampled
        laplacian_pyramid.append(laplacian)

    # The last level is the same as the last level of the Gaussian pyramid
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

def convert_to_tensor(data):
    # Convert Dask array to NumPy array
    if isinstance(data, da.Array):
        data = data.compute()

    # Convert NumPy array to PyTorch tensor
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
        data = data.view(1,1,512,512)

    return data

def kld_loss(hr_image, sr_image, num_levels=3):
    # Apply Gaussian Pyramid
    hr_image = convert_to_tensor(hr_image)
    sr_image = convert_to_tensor(sr_image)
    gauss_sr = gaussian_pyramid(sr_image, num_levels)
    gauss_hr = gaussian_pyramid(hr_image, num_levels)

    # Apply Laplacian Pyramid
    lapl_hr = laplacian_pyramid(gauss_hr)
    lapl_sr = laplacian_pyramid(gauss_sr)

    # Normalize Laplacian Pyramid
    norm_lapl_hr = [normalize_laplacian_level(layer) for layer in lapl_hr]
    norm_lapl_sr = [normalize_laplacian_level(layer) for layer in lapl_sr]

    # Calculate Histograms and KLD for each level
    kld_values = []
    for layer_hr, layer_sr in zip(norm_lapl_hr, norm_lapl_sr):
        hist_hr = calculate_histogram(layer_hr, bins=256, range=(layer_hr.min(), layer_hr.max()))
        hist_sr = calculate_histogram(layer_sr, bins=256, range=(layer_sr.min(), layer_sr.max()))
        kldd = kld(hist_hr, hist_sr)
        kld_values.append(kldd)

    return kld_values[0].item()
