"""
Physics-Informed Neural Network for Perfusion Estimation

Uses the Pennes Bioheat Equation as a physics constraint in the loss:

    ρc * dT/dt = k∇²T + w_b * c_b * (T_a - T) + Q_m

Simplified for perfusion:
    w_b ∝ dT/dt - α∇²T

The network predicts perfusion, and the loss ensures the prediction
is consistent with the observed temperature dynamics via the bioheat equation.

This is a true Physics-Informed Neural Network (PINN) approach!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# Differentiable Bioheat Operations
# ============================================================================

def gaussian_kernel2d(kernel_size: int, sigma: float, device):
    """Create 2D Gaussian kernel."""
    ax = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def smooth2d(x, sigma: float):
    """Differentiable Gaussian smoothing."""
    if sigma <= 0:
        return x
    ksize = int(max(3, round(sigma * 4) | 1))
    k = gaussian_kernel2d(ksize, sigma, x.device).view(1, 1, ksize, ksize)
    padding = ksize // 2
    return F.conv2d(x, k, padding=padding)


def laplacian2d(x):
    """Differentiable Laplacian operator."""
    # x: (B, 1, H, W)
    kernel = torch.tensor([[0.0, 1.0, 0.0],
                           [1.0, -4.0, 1.0],
                           [0.0, 1.0, 0.0]], device=x.device, dtype=x.dtype)
    kernel = kernel.view(1, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1)


def compute_temporal_derivative(frames: torch.Tensor):
    """
    Compute dT/dt from frame sequence using linear regression.
    
    Args:
        frames: (B, T, H, W) tensor of thermal frames
    Returns:
        slope: (B, 1, H, W) temporal derivative
    """
    B, T, H, W = frames.shape
    device = frames.device
    
    # Time indices centered at zero
    t = torch.arange(T, device=device, dtype=frames.dtype)
    t = t - t.mean()  # Center
    t = t.view(1, T, 1, 1)  # (1, T, 1, 1)
    
    # Least squares slope: sum(t * (y - mean(y))) / sum(t^2)
    frames_centered = frames - frames.mean(dim=1, keepdim=True)
    numerator = (t * frames_centered).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    denominator = (t ** 2).sum() + 1e-8
    
    slope = numerator / denominator
    return slope


def bioheat_forward(perfusion: torch.Tensor, 
                    mean_temp: torch.Tensor,
                    alpha: float = 0.00014,
                    smoothing_sigma: float = 2.0,
                    pixel_size_mm: float = 0.5):
    """
    Forward bioheat model: Given perfusion, predict what dT/dt should be.
    
    Bioheat equation (simplified):
        dT/dt = α∇²T + w_b  (where w_b is perfusion)
    
    Args:
        perfusion: (B, 1, H, W) predicted perfusion
        mean_temp: (B, 1, H, W) mean temperature field
        alpha: thermal diffusivity
        smoothing_sigma: for smoothing before Laplacian
        pixel_size_mm: pixel size in mm
    
    Returns:
        predicted_slope: (B, 1, H, W) predicted dT/dt
    """
    # Smooth temperature for Laplacian
    smooth_temp = smooth2d(mean_temp, smoothing_sigma)
    
    # Compute Laplacian (∇²T)
    lap = laplacian2d(smooth_temp) / ((pixel_size_mm / 1000.0) ** 2 + 1e-8)
    
    # Predicted slope from bioheat equation
    # dT/dt = α∇²T + perfusion_source
    predicted_slope = alpha * lap + perfusion
    
    return predicted_slope


# ============================================================================
# Network Architecture
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class PhysicsInformedPerfusionNet(nn.Module):
    """
    Physics-Informed Network for Perfusion Estimation.
    
    Input: Thermal frame statistics (mean, slope, std)
    Output: Perfusion map
    
    The network is trained with physics constraints from the bioheat equation.
    """
    
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        
        bc = base_channels
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, bc)
        self.enc2 = ConvBlock(bc, bc * 2)
        self.enc3 = ConvBlock(bc * 2, bc * 4)
        self.enc4 = ConvBlock(bc * 4, bc * 8)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(bc * 8, bc * 8)
        
        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = ConvBlock(bc * 8 + bc * 8, bc * 4)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(bc * 4 + bc * 4, bc * 2)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(bc * 2 + bc * 2, bc)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(bc + bc, bc)
        
        # Output: perfusion (non-negative via softplus)
        self.out_conv = nn.Conv2d(bc, 1, 1)
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - [mean_temp, slope, std] features
        Returns:
            perfusion: (B, 1, H, W) - predicted perfusion (non-negative)
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Perfusion (non-negative)
        out = self.out_conv(d1)
        perfusion = self.softplus(out)
        
        return perfusion


# ============================================================================
# Physics-Informed Loss
# ============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss combining:
    1. Data loss: Match observed bioheat (supervised)
    2. Physics loss: Consistency with bioheat equation (unsupervised)
    3. Smoothness: Perfusion should be spatially smooth
    """
    
    def __init__(self, 
                 data_weight=1.0, 
                 physics_weight=0.5,
                 smooth_weight=0.1,
                 edge_weight: float = 0.0,
                 alpha=0.00014,
                 pixel_size_mm=0.5):
        super().__init__()
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.smooth_weight = smooth_weight
        self.edge_weight = edge_weight
        self.alpha = alpha
        self.pixel_size_mm = pixel_size_mm

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def smoothness_loss(self, x):
        """Total variation for smoothness."""
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()
    
    def edge_loss(self, pred, target):
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        return F.l1_loss(
            torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-8),
            torch.sqrt(target_gx**2 + target_gy**2 + 1e-8),
        )

    def forward(self, pred_perfusion, target_bioheat, observed_slope, mean_temp, mask=None):
        """
        Args:
            pred_perfusion: (B, 1, H, W) network output
            target_bioheat: (B, 1, H, W) ground truth bioheat map
            observed_slope: (B, 1, H, W) observed dT/dt from frames
            mean_temp: (B, 1, H, W) mean temperature
        """
        # 1. Data loss: Match ground truth bioheat
        if mask is not None:
            data_loss = (torch.abs(pred_perfusion - target_bioheat) * mask).sum() / (mask.sum() + 1e-8)
        else:
            data_loss = F.l1_loss(pred_perfusion, target_bioheat)
        
        # 2. Physics loss: Predicted perfusion should explain observed slope
        # Using bioheat equation: dT/dt = α∇²T + w_b
        predicted_slope = bioheat_forward(
            pred_perfusion, mean_temp,
            alpha=self.alpha, 
            pixel_size_mm=self.pixel_size_mm
        )
        if mask is not None:
            physics_loss = (torch.abs(predicted_slope - observed_slope) * mask).sum() / (mask.sum() + 1e-8)
        else:
            physics_loss = F.l1_loss(predicted_slope, observed_slope)
        
        # 3. Smoothness loss
        smooth_loss = self.smoothness_loss(pred_perfusion)

        # 4. Optional edge loss (crisper outputs)
        if self.edge_weight > 0:
            edge_loss = self.edge_loss(pred_perfusion, target_bioheat)
        else:
            edge_loss = pred_perfusion.new_tensor(0.0)
        
        total = (self.data_weight * data_loss + 
                 self.physics_weight * physics_loss + 
                 self.smooth_weight * smooth_loss +
                 self.edge_weight * edge_loss)
        
        return total, {
            'data': data_loss.item(),
            'physics': physics_loss.item(),
            'smooth': smooth_loss.item(),
            'edge': edge_loss.item(),
        }


# ============================================================================
# Training Function
# ============================================================================

def train_physics_informed(
    model: nn.Module,
    train_data: list,  # List of (frames, target_bioheat, mask) tuples
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda",
    save_path: Path = None,
    early_stop_patience: int = 20,
    alpha: float = 0.00014,
    pixel_size_mm: float = 0.5,
):
    """Train physics-informed perfusion network."""
    
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Multi-GPU support
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using DataParallel with {n_gpus} GPUs")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        model = nn.DataParallel(model)
    
    model = model.to(device_t)
    
    criterion = PhysicsInformedLoss(
        data_weight=1.0,
        physics_weight=0.5,
        smooth_weight=0.1,
        alpha=alpha,
        pixel_size_mm=pixel_size_mm,
    ).to(device_t)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_losses = {'data': 0, 'physics': 0, 'smooth': 0}
        n_samples = 0
        
        indices = np.random.permutation(len(train_data))
        
        pbar = tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for i in pbar:
            batch_idx = indices[i:i+batch_size]
            
            # Collect batch
            frames_list = []
            target_list = []
            for idx in batch_idx:
                frames, target = train_data[idx]
                frames_list.append(frames)
                target_list.append(target)
            
            frames_batch = torch.stack(frames_list).to(device_t)  # (B, T, H, W)
            target_batch = torch.stack(target_list).to(device_t).unsqueeze(1)  # (B, 1, H, W)
            
            B, T, H, W = frames_batch.shape
            
            # Compute temporal features from frames
            # 1. Mean temperature
            mean_temp = frames_batch.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            
            # 2. Observed slope (dT/dt)
            observed_slope = compute_temporal_derivative(frames_batch)  # (B, 1, H, W)
            observed_slope = F.relu(observed_slope)  # Clamp to positive
            
            # 3. Standard deviation (noise indicator)
            std_temp = frames_batch.std(dim=1, keepdim=True)  # (B, 1, H, W)
            
            # Normalize inputs
            mean_norm = (mean_temp - mean_temp.mean()) / (mean_temp.std() + 1e-8)
            slope_norm = (observed_slope - observed_slope.mean()) / (observed_slope.std() + 1e-8)
            std_norm = (std_temp - std_temp.mean()) / (std_temp.std() + 1e-8)
            
            # Stack as input features
            x = torch.cat([mean_norm, slope_norm, std_norm], dim=1)  # (B, 3, H, W)
            
            # Normalize target
            target_norm = (target_batch - target_batch.mean()) / (target_batch.std() + 1e-8)
            
            optimizer.zero_grad()
            
            # Forward
            pred_perfusion = model(x)
            
            # Resize if needed
            if pred_perfusion.shape[-2:] != target_batch.shape[-2:]:
                pred_perfusion = F.interpolate(pred_perfusion, size=target_batch.shape[-2:],
                                               mode='bilinear', align_corners=True)
            
            # Normalize prediction for loss
            pred_norm = (pred_perfusion - pred_perfusion.mean()) / (pred_perfusion.std() + 1e-8)
            
            # Physics-informed loss
            loss, loss_components = criterion(
                pred_norm, target_norm, 
                slope_norm, mean_norm
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * B
            for k, v in loss_components.items():
                epoch_losses[k] += v * B
            n_samples += B
            pbar.set_postfix(loss=loss.item())
        
        epoch_loss /= n_samples
        for k in epoch_losses:
            epoch_losses[k] /= n_samples
        loss_history.append(epoch_loss)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: loss={epoch_loss:.6f} "
              f"(data={epoch_losses['data']:.4f}, physics={epoch_losses['physics']:.4f}, "
              f"smooth={epoch_losses['smooth']:.4f}), lr={optimizer.param_groups[0]['lr']:.2e}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            if save_path:
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss,
                    'loss_components': epoch_losses,
                    'n_gpus': n_gpus,
                }, save_path)
                print(f"  Saved best model (loss={best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if isinstance(model, nn.DataParallel):
        model = model.module
    return model, loss_history
