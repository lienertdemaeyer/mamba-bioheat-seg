import torch
import torch.nn.functional as F


def gaussian_kernel2d(kernel_size: int, sigma: float, device):
    ax = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def smooth2d(x, sigma: float):
    if sigma <= 0:
        return x
    ksize = int(max(3, round(sigma * 4) | 1))  # odd
    k = gaussian_kernel2d(ksize, sigma, x.device).view(1, 1, ksize, ksize)
    padding = ksize // 2
    return F.conv2d(x, k, padding=padding)


def laplace2d(x):
    # x: (N,1,H,W)
    kernel = torch.tensor([[0.0, 1.0, 0.0],
                           [1.0, -4.0, 1.0],
                           [0.0, 1.0, 0.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1)


def compute_slope(frames: torch.Tensor, start: int, end: int):
    # frames: (T, H, W)
    end = min(end, frames.shape[0])
    start = max(0, start)
    window = frames[start:end]  # (t, h, w)
    if window.shape[0] < 2:
        return torch.zeros_like(frames[0])
    t = torch.arange(window.shape[0], device=frames.device, dtype=frames.dtype)
    t = t - t.mean()
    denom = torch.sum(t * t) + 1e-8
    mean_w = window.mean(dim=0)
    slope = torch.tensordot(t, (window - mean_w), dims=1) / denom
    slope = torch.clamp(slope, min=0)
    return slope


def compute_bioheat(frames: torch.Tensor, mask: torch.Tensor, start: int, end: int,
                    smoothing_sigma: float, pixel_size_mm: float, alpha: float):
    # frames: (T,H,W) float32
    slope = compute_slope(frames, start, end)
    mean_frame = frames[start:end].mean(dim=0)
    smooth_frame = smooth2d(mean_frame[None, None, :, :], smoothing_sigma)[0, 0]
    lap = laplace2d(smooth_frame[None, None, :, :])[0, 0] / ((pixel_size_mm / 1000.0) ** 2 + 1e-8)
    source = slope - alpha * lap
    source = torch.clamp(source, min=0)
    source = source * mask
    return source
