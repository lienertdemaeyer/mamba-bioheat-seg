from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from tqdm import tqdm
import cv2

from .bioheat_torch import compute_bioheat
from .data_io import load_frames_h5, load_focus, load_mask
from .graphcut import segment_graphcut_contrast


def get_mm_per_pixel(focus_mm, w=640, h=480):
    alpha_x = np.radians(12.5)
    alpha_y = np.radians(9.5)
    fov_x_mm = 2 * focus_mm * np.tan(alpha_x)
    fov_y_mm = 2 * focus_mm * np.tan(alpha_y)
    return ((fov_x_mm / w) + (fov_y_mm / h)) / 2


def fuse_sequence(
    h5_dir: Path,
    mask_path: Path,
    patient: str,
    measurements: List[str],
    window_size: int,
    window_step: int,
    start_at: int,
    end_at: int,
    fuse_mode: str,
    fuse_percentile: float,
    fuse_neighbor: int,
    binary_percentile: float,
    mask_erode_px: int,
    device: str,
    segmentation_mode: str = "percentile",
    graphcut_lambda: float = 10.0,
    graphcut_fg_pct: float = 85.0,
    graphcut_bg_pct: float = 30.0,
    graphcut_gamma: float = 1.5,
    graphcut_min_component_px: int = 50,
    skip_pids: List[str] = None,
    late_only_fusion: bool = False,
    include_partial_cooled: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    focus_mm = load_focus(h5_dir, patient)
    skip_pids = skip_pids or []
    results = {}
    for meas in measurements:
        pid = f"{patient}{meas}"
        if pid in skip_pids:
            print(f"[skip] {pid}: in skip list")
            continue
        frames_np = load_frames_h5(h5_dir, pid)
        if frames_np is None:
            continue
        
        h, w = frames_np.shape[1], frames_np.shape[2]
        mm_per_px = get_mm_per_pixel(focus_mm, w, h)
        mask_np = load_mask(pid, (h, w), mask_path, include_partial=include_partial_cooled)
        if mask_erode_px > 0:
            mask_np = cv2.erode(mask_np.astype(np.uint8), np.ones((mask_erode_px, mask_erode_px), np.uint8)).astype(bool)
        frames_t = torch.from_numpy(frames_np).to(device_t)
        mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(device_t)

        n_frames = frames_np.shape[0]
        start_min = max(0, start_at)
        start_max = n_frames - window_size
        if end_at is not None:
            start_max = min(start_max, end_at - window_size)
        starts = list(range(start_min, start_max + 1, window_step)) if start_max >= start_min else []
        if not starts:
            continue

        window_maps = []
        for s in tqdm(starts, desc=f"{pid} windows", unit="win"):
            e = s + window_size
            bh = compute_bioheat(
                frames_t, mask_t, s, e,
                smoothing_sigma=2.0,
                pixel_size_mm=mm_per_px,
                alpha=0.00014,
            )
            window_maps.append(bh)

        stack = torch.stack(window_maps, dim=0)  # (num_win, H, W)
        
        # Late-only fusion - use last 50% of windows to suppress early artifacts
        use_late_only = late_only_fusion
        
        # Fuse neighboring windows
        fused_windows = []
        for i in range(stack.shape[0]):
            lo = max(0, i - fuse_neighbor)
            hi = min(stack.shape[0] - 1, i + fuse_neighbor)
            sub = stack[lo:hi+1]
            if fuse_mode == "percentile":
                fused = torch.quantile(sub, fuse_percentile / 100.0, dim=0)
            else:
                fused = torch.median(sub, dim=0).values
            fused_windows.append(fused)
        fused_stack = torch.stack(fused_windows, dim=0)
        
        # Final global fusion - use late-only when late_only_fusion is enabled
        if use_late_only:
            late_start = fused_stack.shape[0] // 2
            fusion_stack = fused_stack[late_start:]
            window_range = f"{late_start}-{fused_stack.shape[0]-1}"
        else:
            fusion_stack = fused_stack
            window_range = f"0-{fused_stack.shape[0]-1}"
        
        # Global fusion method: median, mean, or percentile
        if fuse_mode == "mean":
            global_fused = fusion_stack.mean(dim=0)
            fusion_method = "mean"
        elif fuse_mode == "percentile":
            # Use fuse_percentile for final fusion too (e.g., 30th percentile to suppress high artifacts)
            global_fused = torch.quantile(fusion_stack, fuse_percentile / 100.0, dim=0)
            fusion_method = f"p{fuse_percentile:.0f}"
        else:  # median (default)
            global_fused = torch.median(fusion_stack, dim=0).values
            fusion_method = "median"
        
        # Debug output
        if use_late_only:
            early_stack = fused_stack[:late_start]
            print(f"  [{fusion_method}] {pid}: late-only windows {window_range}")
            print(f"    early mean={early_stack.mean().item():.4f}, max={early_stack.max().item():.4f}")
            print(f"    late  mean={fusion_stack.mean().item():.4f}, max={fusion_stack.max().item():.4f}")
        else:
            print(f"  [{fusion_method}] {pid}: all windows {window_range}")
        global_fused_np = global_fused.detach().cpu().numpy()
        mask_np = mask_np.astype(bool)
        
        if segmentation_mode == "graphcut":
            binary = segment_graphcut_contrast(
                global_fused_np,
                mask_np.astype(np.uint8),
                lam=graphcut_lambda,
                fg_pct=graphcut_fg_pct,
                bg_pct=graphcut_bg_pct,
                gamma=graphcut_gamma,
                min_component_px=graphcut_min_component_px,
            )
            thr = None  # graph-cut doesn't use a single threshold
        else:
            # Default: percentile thresholding
            valid = global_fused_np[mask_np & (global_fused_np > 0)]
            thr = np.percentile(valid, binary_percentile) if valid.size > 0 else 0.0
            binary = ((global_fused_np >= thr) & mask_np).astype(np.uint8)
        
        results[meas] = {
            "fused": global_fused_np,
            "binary": binary.astype(np.uint8),
            "threshold": thr,
            "starts": starts,
            "mask": mask_np.astype(np.uint8),
            "segmentation": segmentation_mode,
            "late_only_fusion": use_late_only,
        }
    return results
