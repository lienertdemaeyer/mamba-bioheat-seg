"""
TRUE End-to-End 3D Network: Raw Frames → Perfusion

Uses 3D convolutions to learn directly from raw thermal frame sequences.
No pre-computed statistics - the network learns the temporal patterns!

This is the most novel approach - learning the physics from data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# 3D Convolutional Blocks
# ============================================================================

class Conv3DBlock(nn.Module):
    """3D convolution block with instance norm."""
    
    def __init__(self, in_ch, out_ch, temporal_kernel=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 
                      kernel_size=(temporal_kernel, 3, 3),
                      padding=(temporal_kernel//2, 1, 1)),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch,
                      kernel_size=(temporal_kernel, 3, 3),
                      padding=(temporal_kernel//2, 1, 1)),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class Conv2DBlock(nn.Module):
    """2D convolution block for spatial processing."""
    
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


# ============================================================================
# 3D Temporal Encoder
# ============================================================================

class TemporalEncoder3D(nn.Module):
    """
    3D encoder that processes raw frame sequences.
    Progressively reduces temporal dimension while extracting features.
    """
    
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        
        bc = base_channels
        
        # 3D encoder layers - reduce temporal and spatial dims
        self.enc1 = Conv3DBlock(in_channels, bc, temporal_kernel=5)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # T/2, H/2, W/2
        
        self.enc2 = Conv3DBlock(bc, bc * 2, temporal_kernel=5)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # T/4, H/4, W/4
        
        self.enc3 = Conv3DBlock(bc * 2, bc * 4, temporal_kernel=3)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # T/8, H/8, W/8
        
        self.enc4 = Conv3DBlock(bc * 4, bc * 8, temporal_kernel=3)
        
        # Temporal aggregation - collapse time dimension
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))  # (B, C, 1, H, W)
        
    def forward(self, x):
        """
        Args:
            x: (B, 1, T, H, W) raw thermal frames
        Returns:
            features: (B, C, H/8, W/8) spatial features
            skip_connections: list of encoder features for decoder
        """
        # 3D encoding
        e1 = self.enc1(x)           # (B, bc, T, H, W)
        p1 = self.pool1(e1)         # (B, bc, T/2, H/2, W/2)
        
        e2 = self.enc2(p1)          # (B, bc*2, T/2, H/2, W/2)
        p2 = self.pool2(e2)         # (B, bc*2, T/4, H/4, W/4)
        
        e3 = self.enc3(p2)          # (B, bc*4, T/4, H/4, W/4)
        p3 = self.pool3(e3)         # (B, bc*4, T/8, H/8, W/8)
        
        e4 = self.enc4(p3)          # (B, bc*8, T/8, H/8, W/8)
        
        # Collapse temporal dimension
        features = self.temporal_pool(e4).squeeze(2)  # (B, bc*8, H/8, W/8)
        
        # Also collapse skip connections
        s1 = self.temporal_pool(e1).squeeze(2)  # (B, bc, H, W)
        s2 = self.temporal_pool(e2).squeeze(2)  # (B, bc*2, H/2, W/2)
        s3 = self.temporal_pool(e3).squeeze(2)  # (B, bc*4, H/4, W/4)
        
        return features, [s1, s2, s3]


# ============================================================================
# 2D Spatial Decoder
# ============================================================================

class SpatialDecoder2D(nn.Module):
    """2D decoder with skip connections from encoder."""
    
    def __init__(self, base_channels=16):
        super().__init__()
        
        bc = base_channels
        
        # Bottleneck
        self.bottleneck = Conv2DBlock(bc * 8, bc * 8)
        
        # Decoder with skip connections
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2DBlock(bc * 8 + bc * 4, bc * 4)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2DBlock(bc * 4 + bc * 2, bc * 2)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2DBlock(bc * 2 + bc, bc)
        
        # Output
        self.out_conv = nn.Conv2d(bc, 1, 1)
    
    def forward(self, features, skips):
        """
        Args:
            features: (B, C, H/8, W/8) bottleneck features
            skips: [s1, s2, s3] skip connections from encoder
        Returns:
            output: (B, 1, H, W) perfusion map
        """
        s1, s2, s3 = skips
        
        # Bottleneck
        x = self.bottleneck(features)
        
        # Decoder
        x = self.up3(x)
        x = self.dec3(torch.cat([x, s3], dim=1))
        
        x = self.up2(x)
        x = self.dec2(torch.cat([x, s2], dim=1))
        
        x = self.up1(x)
        x = self.dec1(torch.cat([x, s1], dim=1))
        
        out = self.out_conv(x)
        
        return out


# ============================================================================
# Full 3D End-to-End Network
# ============================================================================

class TrueEndToEnd3D(nn.Module):
    """
    TRUE End-to-End network using 3D convolutions on raw frame sequences.
    
    Input: (B, 1, T, H, W) - T raw thermal frames
    Output: (B, 1, H, W) - Perfusion map
    
    The network learns:
    1. What temporal patterns are relevant (via 3D convs)
    2. How to aggregate temporal information
    3. How to generate a perfusion map
    
    No hand-crafted features - everything is learned!
    """
    
    def __init__(self, base_channels=16):
        super().__init__()
        
        self.encoder = TemporalEncoder3D(in_channels=1, base_channels=base_channels)
        self.decoder = SpatialDecoder2D(base_channels=base_channels)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, H, W) or (B, 1, T, H, W) raw thermal frames
        Returns:
            (B, 1, H, W) perfusion map
        """
        # Add channel dim if needed
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, T, H, W) -> (B, 1, T, H, W)
        
        # Encode with 3D convolutions
        features, skips = self.encoder(x)
        
        # Decode with 2D convolutions
        out = self.decoder(features, skips)
        
        return out


# ============================================================================
# Physics-Informed Loss Function
# ============================================================================

def compute_temporal_derivative(frames):
    """
    Compute dT/dt from frame sequence.
    frames: (B, T, H, W)
    returns: (B, 1, H, W)
    """
    B, T, H, W = frames.shape
    t = torch.arange(T, device=frames.device, dtype=frames.dtype)
    t = t - t.mean()
    t = t.view(1, T, 1, 1)
    
    frames_centered = frames - frames.mean(dim=1, keepdim=True)
    numerator = (t * frames_centered).sum(dim=1, keepdim=True)
    denominator = (t ** 2).sum() + 1e-8
    
    slope = numerator / denominator
    return F.relu(slope)  # Temperature should rise during cooling


def compute_laplacian(x):
    """
    Compute Laplacian ∇²x.
    x: (B, 1, H, W)
    returns: (B, 1, H, W)
    """
    kernel = torch.tensor([[0.0, 1.0, 0.0],
                           [1.0, -4.0, 1.0],
                           [0.0, 1.0, 0.0]], device=x.device, dtype=x.dtype)
    kernel = kernel.view(1, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1)


class PhysicsInformedE2ELoss(nn.Module):
    """
    Physics-Informed Loss for True End-to-End learning.
    
    Components:
    1. Data loss: Match target bioheat map
    2. Physics loss: Predicted perfusion must satisfy bioheat equation!
       dT/dt = α∇²T + perfusion
    3. SSIM loss: Structure preservation
    4. Smoothness: Regularization
    
    The physics loss ensures hotspots are correctly identified:
    - High dT/dt regions → must have high predicted perfusion
    - This enforces physically consistent learning!
    """
    
    def __init__(self, 
                 data_weight=1.0, 
                 physics_weight=0.5,  # NEW: Physics constraint!
                 ssim_weight=0.3, 
                 smooth_weight=0.1,
                 alpha=0.00014,  # Thermal diffusivity
                 pixel_size_mm=0.5):
        super().__init__()
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.ssim_weight = ssim_weight
        self.smooth_weight = smooth_weight
        self.alpha = alpha
        self.pixel_size_mm = pixel_size_mm
    
    def ssim_loss(self, pred, target, window_size=11):
        """SSIM loss for structure preservation."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        sigma_pred = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred ** 2
        sigma_target = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target ** 2
        sigma_both = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred * mu_target
        
        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_both + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
        
        return 1 - ssim.mean()
    
    def smoothness_loss(self, x):
        """Total variation for spatial smoothness."""
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()
    
    def physics_loss(self, pred_perfusion, frames, mean_temp):
        """
        Physics loss: Ensure predicted perfusion satisfies bioheat equation.
        
        Bioheat equation: dT/dt = α∇²T + perfusion
        
        So: perfusion = dT/dt - α∇²T
        
        The predicted perfusion should match this!
        """
        # Compute observed dT/dt from frames
        observed_slope = compute_temporal_derivative(frames)  # (B, 1, H, W)
        
        # Compute Laplacian of mean temperature
        laplacian = compute_laplacian(mean_temp)
        laplacian = laplacian / ((self.pixel_size_mm / 1000.0) ** 2 + 1e-8)
        
        # Physics-derived perfusion: what perfusion SHOULD be
        physics_perfusion = observed_slope - self.alpha * laplacian
        physics_perfusion = F.relu(physics_perfusion)  # Non-negative
        
        # Normalize for comparison
        pred_norm = (pred_perfusion - pred_perfusion.mean()) / (pred_perfusion.std() + 1e-8)
        physics_norm = (physics_perfusion - physics_perfusion.mean()) / (physics_perfusion.std() + 1e-8)
        
        return F.l1_loss(pred_norm, physics_norm)
    
    def forward(self, pred, target, frames=None, mean_temp=None):
        """
        Args:
            pred: (B, 1, H, W) predicted perfusion
            target: (B, 1, H, W) target bioheat map
            frames: (B, T, H, W) raw frames for physics loss
            mean_temp: (B, 1, H, W) mean temperature for physics loss
        """
        losses = {}
        
        # 1. Data loss
        data_loss = F.l1_loss(pred, target)
        losses['data'] = data_loss.item()
        
        # 2. SSIM loss
        ssim = self.ssim_loss(pred, target)
        losses['ssim'] = ssim.item()
        
        # 3. Smoothness loss
        smooth = self.smoothness_loss(pred)
        losses['smooth'] = smooth.item()
        
        # 4. Physics loss (if frames provided)
        if frames is not None and mean_temp is not None:
            physics = self.physics_loss(pred, frames, mean_temp)
            losses['physics'] = physics.item()
        else:
            physics = torch.tensor(0.0, device=pred.device)
            losses['physics'] = 0.0
        
        total = (self.data_weight * data_loss + 
                 self.physics_weight * physics +
                 self.ssim_weight * ssim + 
                 self.smooth_weight * smooth)
        
        return total, losses


# Keep simple version for backward compatibility
class TrueE2ELoss(nn.Module):
    """Simple combined loss (no physics)."""
    
    def __init__(self, l1_weight=1.0, ssim_weight=0.3, smooth_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.smooth_weight = smooth_weight
    
    def ssim_loss(self, pred, target, window_size=11):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        sigma_pred = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred ** 2
        sigma_target = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target ** 2
        sigma_both = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred * mu_target
        
        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_both + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
        
        return 1 - ssim.mean()
    
    def smoothness_loss(self, x):
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()
    
    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        smooth = self.smoothness_loss(pred)
        
        return self.l1_weight * l1 + self.ssim_weight * ssim + self.smooth_weight * smooth


# ============================================================================
# Training Function
# ============================================================================

def train_true_e2e(
    model: nn.Module,
    train_data: list,  # List of (frames, target) tuples
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-4,
    device: str = "cuda",
    save_path: Path = None,
    early_stop_patience: int = 20,
    use_physics_loss: bool = True,  # NEW: Enable physics loss!
):
    """Train true end-to-end 3D network with DataParallel and physics loss."""
    
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Multi-GPU support
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using DataParallel with {n_gpus} GPUs")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        model = nn.DataParallel(model)
    
    model = model.to(device_t)
    
    # Use physics-informed loss!
    if use_physics_loss:
        print("Using Physics-Informed Loss (bioheat equation constraint)")
        criterion = PhysicsInformedE2ELoss(
            data_weight=1.0,
            physics_weight=0.5,
            ssim_weight=0.3,
            smooth_weight=0.1,
        ).to(device_t)
    else:
        criterion = TrueE2ELoss().to(device_t)
    
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
            target_batch = torch.stack(target_list).to(device_t)  # (B, H, W)
            
            # Add channel dim to target
            if target_batch.dim() == 3:
                target_batch = target_batch.unsqueeze(1)  # (B, 1, H, W)
            
            B = frames_batch.shape[0]
            
            # Normalize frames per-sample
            frames_flat = frames_batch.view(B, -1)
            f_mean = frames_flat.mean(dim=1, keepdim=True).view(B, 1, 1, 1)
            f_std = frames_flat.std(dim=1, keepdim=True).view(B, 1, 1, 1) + 1e-8
            frames_norm = (frames_batch - f_mean) / f_std
            
            # Normalize target per-sample
            target_flat = target_batch.view(B, -1)
            t_mean = target_flat.mean(dim=1, keepdim=True).view(B, 1, 1, 1)
            t_std = target_flat.std(dim=1, keepdim=True).view(B, 1, 1, 1) + 1e-8
            target_norm = (target_batch - t_mean) / t_std
            
            optimizer.zero_grad()
            
            # Forward - add channel dim for 3D conv
            pred = model(frames_norm.unsqueeze(1))  # (B, 1, T, H, W)
            
            # Resize if needed
            if pred.shape[-2:] != target_batch.shape[-2:]:
                pred = F.interpolate(pred, size=target_batch.shape[-2:],
                                    mode='bilinear', align_corners=True)
            
            # Compute loss (with physics if enabled)
            if use_physics_loss:
                # Compute mean temperature for physics loss
                mean_temp = frames_batch.mean(dim=1, keepdim=True)  # (B, 1, H, W)
                mean_temp_norm = (mean_temp - mean_temp.mean()) / (mean_temp.std() + 1e-8)
                
                loss, loss_dict = criterion(pred, target_norm, frames_norm, mean_temp_norm)
            else:
                loss = criterion(pred, target_norm)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * B
            n_samples += B
            pbar.set_postfix(loss=loss.item())
        
        epoch_loss /= n_samples
        loss_history.append(epoch_loss)
        scheduler.step()
        
        if use_physics_loss:
            print(f"Epoch {epoch+1}: loss={epoch_loss:.6f} (physics-informed), lr={optimizer.param_groups[0]['lr']:.2e}")
        else:
            print(f"Epoch {epoch+1}: loss={epoch_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            if save_path:
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss,
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
