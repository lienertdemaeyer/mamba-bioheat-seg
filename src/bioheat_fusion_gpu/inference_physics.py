"""
TRUE E2E inference + visualization pipeline.

Loads the TRUE E2E 3D network, applies the cooled area mask correctly,
and recreates the diagnostic plots we used for earlier end-to-end runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Model + data helpers
# -----------------------------------------------------------------------------

def load_true_e2e_model(model_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load the TRUE E2E 3D checkpoint."""
    from .true_e2e_3d import TrueEndToEnd3D

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    base_channels = state["encoder.enc1.conv.0.weight"].shape[0]

    model = TrueEndToEnd3D(base_channels=base_channels).to(device)
    model.load_state_dict(state)
    model.eval()

    print(f"Loaded TRUE E2E model: {model_path}")
    epoch = checkpoint.get("epoch")
    loss = checkpoint.get("loss")
    if epoch is not None:
        print(f"  Epoch: {epoch}")
    if isinstance(loss, (int, float)):
        print(f"  Loss: {loss:.6f}")

    return model


def _read_first_available(dataset: h5py.File, candidate_paths: List[str]) -> Optional[np.ndarray]:
    for path in candidate_paths:
        if path in dataset:
            return dataset[path][:]
    return None


def load_frames_from_h5(h5_path: str, measurement: str) -> Optional[np.ndarray]:
    """Load thermal frames (T, H, W)."""
    candidate_paths = [
        f"Measurements/Cooling/{measurement}/frames",
        f"measurements/{measurement}/frames",
        f"{measurement}/frames",
        f"{measurement}/thermal_frames",
        f"measurements/{measurement}/thermal_frames",
    ]
    try:
        with h5py.File(h5_path, "r") as f:
            frames = _read_first_available(f, candidate_paths)
            if frames is not None:
                print(f"    Frames loaded: {frames.shape}")
            return frames
    except Exception as exc:  # pragma: no cover - limited to runtime I/O
        print(f"    Failed to load frames from {h5_path}: {exc}")
        return None


def load_target_from_h5(h5_path: str, measurement: str) -> Optional[np.ndarray]:
    """Load target bioheat map if it is stored in the H5 file."""
    candidate_paths = [
        f"Measurements/Cooling/{measurement}/bioheat",
        f"measurements/{measurement}/bioheat",
        f"{measurement}/bioheat",
        f"Measurements/Cooling/{measurement}/target",
        f"measurements/{measurement}/target",
        f"{measurement}/target",
    ]
    try:
        with h5py.File(h5_path, "r") as f:
            target = _read_first_available(f, candidate_paths)
            if target is not None:
                print(f"    Bioheat target loaded: {target.shape}")
            return target
    except Exception as exc:  # pragma: no cover - limited to runtime I/O
        print(f"    Failed to load target from {h5_path}: {exc}")
        return None


def compute_bioheat_from_late_frames(frames: np.ndarray, late_frac: float = 0.3) -> np.ndarray:
    """Fallback: compute bioheat from the late portion of the thermal sequence."""
    T = frames.shape[0]
    late_start = max(int(T * (1 - late_frac)), 0)
    late_frames = torch.from_numpy(frames[late_start:].astype(np.float32))

    t = torch.arange(late_frames.shape[0], dtype=torch.float32)
    t = t - t.mean()
    t = t.view(-1, 1, 1)

    frames_centered = late_frames - late_frames.mean(dim=0, keepdim=True)
    numerator = (t * frames_centered).sum(dim=0)
    denominator = (t**2).sum() + 1e-8
    slope = numerator / denominator

    return F.relu(slope).numpy()


def load_mask_from_coco(
    mask_path: Optional[Path], patient_id: str, measurement: str, shape: Tuple[int, int]
) -> np.ndarray:
    """Load binary mask for cooled area."""
    if mask_path is None or not mask_path.exists():
        print("    Mask file missing – defaulting to full image mask.")
        return np.ones(shape, dtype=bool)

    try:
        with open(mask_path, "r") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - limited to runtime I/O
        print(f"    Failed to read mask file ({mask_path}): {exc}")
        return np.ones(shape, dtype=bool)

    pid = f"{patient_id}{measurement}".lower()
    mask = np.zeros(shape, dtype=np.uint8)
    image_id = None
    for img in data.get("images", []):
        if pid in img.get("file_name", "").lower():
            image_id = img.get("id")
            break

    if image_id is None:
        print(f"    Mask not found for {patient_id}/{measurement}; using full mask.")
        return np.ones(shape, dtype=bool)

    for ann in data.get("annotations", []):
        if ann.get("image_id") == image_id and ann.get("segmentation"):
            for seg in ann["segmentation"]:
                pts = np.array(seg, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [pts], 1)

    if mask.sum() == 0:
        print(f"    Empty mask for {patient_id}/{measurement}; using full mask.")
        return np.ones(shape, dtype=bool)

    return mask.astype(bool)


def prepare_frames_for_e2e(
    frames: np.ndarray,
    num_frames: int = 50,
    target_size: Tuple[int, int] = (240, 320),
) -> torch.Tensor:
    """
    Prepare input tensor for the TRUE E2E network.

    Returns (1, T, H, W) tensor.
    """
    frames_t = torch.from_numpy(frames.astype(np.float32))
    T = frames_t.shape[0]

    if T > num_frames:
        idx = torch.linspace(0, T - 1, num_frames).long()
        frames_t = frames_t[idx]
    elif T < num_frames:
        pad = num_frames - T
        frames_t = torch.cat([frames_t, frames_t[-1:].repeat(pad, 1, 1)], dim=0)

    if frames_t.shape[1:] != target_size:
        frames_t = F.interpolate(
            frames_t.unsqueeze(1),
            size=target_size,
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)

    frames_t = frames_t.unsqueeze(0)  # (1, T, H, W)
    frames_t = (frames_t - frames_t.mean()) / (frames_t.std() + 1e-8)
    return frames_t


def resize_to_shape(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resize array to (H, W) using bilinear interpolation."""
    if arr.shape == shape:
        return arr
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=shape, mode="bilinear", align_corners=True)
    return tensor.squeeze().numpy()


def run_true_e2e_inference(
    model: torch.nn.Module,
    frames: np.ndarray,
    device: str,
    num_frames: int,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Run TRUE E2E inference and return numpy map."""
    model = model.to(device)
    x = prepare_frames_for_e2e(frames, num_frames=num_frames, target_size=target_size).to(device)
    with torch.no_grad():
        pred = model(x)
    return pred.squeeze().cpu().numpy()


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def normalize_with_mask(arr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Normalize array to [0, 1] inside the mask."""
    masked_vals = arr[mask]
    if masked_vals.size == 0:
        return np.zeros_like(arr), (0.0, 0.0)
    min_val = float(masked_vals.min())
    max_val = float(masked_vals.max())
    scale = max(max_val - min_val, 1e-8)
    norm = (arr - min_val) / scale
    norm = np.clip(norm, 0.0, 1.0)
    norm[~mask] = np.nan
    return norm, (min_val, max_val)


def visualize_true_e2e(
    frames: np.ndarray,
    target: np.ndarray,
    perfusion: np.ndarray,
    mask: np.ndarray,
    patient_id: str,
    measurement: str,
    output_dir: Path,
    clip_percentile: float = 99.0,
) -> Path:
    """Create 2x3 comparison plot to match prior TRUE E2E diagnostics."""
    target_masked = np.where(mask, target, np.nan)
    perf_masked = np.where(mask, perfusion, np.nan)

    vmax_target = np.nanpercentile(target_masked, clip_percentile)
    vmax_perf = np.nanpercentile(perf_masked, clip_percentile)

    target_norm, (t_min, t_max) = normalize_with_mask(target, mask)
    perf_norm, (p_min, p_max) = normalize_with_mask(perfusion, mask)

    diff_norm = perf_norm - target_norm
    diff_norm[~mask] = np.nan
    vmax_diff = np.nanmax(np.abs(diff_norm))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    im = ax.imshow(target_masked, cmap="hot", vmin=0, vmax=vmax_target)
    ax.set_title(f"Target Bioheat\nmasked 0-{vmax_target:.3f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(perf_masked, cmap="hot", vmin=0, vmax=vmax_perf)
    ax.set_title(f"TRUE E2E 3D (clipped {clip_percentile:.1f}%)\nrange 0-{vmax_perf:.3f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    mean_frame_full = frames.mean(axis=0)
    base = ax.imshow(mean_frame_full, cmap="hot")
    ax.imshow(mask, cmap="gray", alpha=0.35)
    ax.set_title("Mask overlay on mean frame")
    ax.axis("off")
    plt.colorbar(base, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    im = ax.imshow(target_norm, cmap="hot", vmin=0, vmax=1)
    ax.set_title(f"Target normalized 0-1\nmin={t_min:.3f}, max={t_max:.3f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    im = ax.imshow(perf_norm, cmap="hot", vmin=0, vmax=1)
    ax.set_title(f"E2E normalized 0-1\nmin={p_min:.3f}, max={p_max:.3f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    im = ax.imshow(diff_norm, cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff)
    ax.set_title("Normalized difference\n(E2E - Target)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"{patient_id}/{measurement} – TRUE E2E Detailed Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"true_e2e_{patient_id}_{measurement}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"    Saved visualization: {save_path}")
    return save_path


# -----------------------------------------------------------------------------
# Inference loop
# -----------------------------------------------------------------------------

def run_inference_on_patient(
    model: torch.nn.Module,
    h5_path: Path,
    measurements: List[str],
    mask_path: Optional[Path],
    output_dir: Path,
    device: str,
    num_frames: int,
    target_size: Tuple[int, int],
    clip_percentile: float,
):
    patient_id = h5_path.stem
    print(f"\nProcessing patient {patient_id}...")

    for measurement in measurements:
        print(f"  Measurement {measurement}:")
        frames = load_frames_from_h5(str(h5_path), measurement)
        if frames is None:
            print("    Skipping (frames missing).")
            continue

        original_shape = frames.shape[1:]

        target = load_target_from_h5(str(h5_path), measurement)
        if target is None:
            target = compute_bioheat_from_late_frames(frames)
            print("    Target missing, computed from late frames.")

        if target.shape != original_shape:
            target = resize_to_shape(target, original_shape)

        perf = run_true_e2e_inference(
            model,
            frames,
            device=device,
            num_frames=num_frames,
            target_size=target_size,
        )
        if perf.shape != original_shape:
            perf = resize_to_shape(perf, original_shape)

        mask = load_mask_from_coco(mask_path, patient_id, measurement, original_shape)
        mask = mask.astype(bool)

        save_path = visualize_true_e2e(
            frames=frames,
            target=target,
            perfusion=perf,
            mask=mask,
            patient_id=patient_id,
            measurement=measurement,
            output_dir=output_dir,
            clip_percentile=clip_percentile,
        )

        masked_vals_pred = perf[mask]
        masked_vals_target = target[mask]
        print(f"    TRUE E2E stats (masked): min={masked_vals_pred.min():.4f}, "
              f"max={masked_vals_pred.max():.4f}, mean={masked_vals_pred.mean():.4f}")
        print(f"    Target stats (masked):   min={masked_vals_target.min():.4f}, "
              f"max={masked_vals_target.max():.4f}, mean={masked_vals_target.mean():.4f}")
        print(f"    Figure saved at {save_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TRUE E2E inference with masking + diagnostics.")
    parser.add_argument(
        "--model",
        type=str,
        default="/dodrio/scratch/projects/starting_2025_090/output_n2n/models/true_e2e_3d_2gpu.pth",
        help="Path to TRUE E2E checkpoint.",
    )
    parser.add_argument(
        "--h5_dir",
        type=str,
        default="/dodrio/scratch/projects/starting_2025_090/H5",
        help="Directory containing patient H5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/dodrio/scratch/projects/starting_2025_090/output_n2n/true_e2e_inference",
        help="Directory for rendered figures.",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="/dodrio/scratch/projects/starting_2025_090/Annotations/coco_annotations_cooled_area.json",
        help="COCO JSON with cooled area masks.",
    )
    parser.add_argument(
        "--patients",
        type=str,
        nargs="+",
        default=["P18", "P19", "P20"],
        help="Patient IDs to run.",
    )
    parser.add_argument(
        "--measurements",
        type=str,
        nargs="+",
        default=["M01", "M02", "M03", "M04"],
        help="Measurement IDs per patient.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=50,
        help="Number of frames to feed the TRUE E2E model.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=(240, 320),
        help="Spatial size expected by TRUE E2E (H W).",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.0,
        help="Percentile used to clip raw visualizations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on.",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    output_dir = Path(args.output_dir)
    mask_path = Path(args.mask_path) if args.mask_path else None
    h5_dir = Path(args.h5_dir)

    print("=" * 70)
    print("TRUE E2E 3D – Detailed Inference")
    print("=" * 70)
    model = load_true_e2e_model(args.model, device=device)

    for patient_id in args.patients:
        patient_h5 = h5_dir / f"{patient_id}.h5"
        if not patient_h5.exists():
            print(f"\nSkipping {patient_id}: {patient_h5} not found.")
            continue
        run_inference_on_patient(
            model=model,
            h5_path=patient_h5,
            measurements=args.measurements,
            mask_path=mask_path,
            output_dir=output_dir,
            device=device,
            num_frames=args.num_frames,
            target_size=tuple(args.target_size),
            clip_percentile=args.clip_percentile,
        )

    print("\nInference complete. Figures stored in:", output_dir)


if __name__ == "__main__":
    main()
