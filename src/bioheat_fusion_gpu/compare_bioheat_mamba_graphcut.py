"""
Generate comparison figures:
- Grayscale mean frame (no mask)
- Bioheat (fused from windows) + graphcut binary
- Mamba output + graphcut binary

Matches the layout used by `compare_e2e_bioheat_graphcut.py`, but replaces the
TRUE E2E map with a Mamba model prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .bioheat_torch import compute_bioheat
from .data_io import load_frames_h5, load_focus, load_mask
from .fuse import get_mm_per_pixel
from .graphcut import segment_graphcut_contrast
from .inference_mamba import load_mamba_model, run_mamba_inference
from .inference_physics import resize_to_shape


def _to_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        return torch.device("cpu")
    return torch.device(device)


def compute_bioheat_fused(
    frames_t: torch.Tensor,
    mask_t: torch.Tensor,
    mm_per_px: float,
    window_size: int,
    window_step: int,
    late_only_fusion: bool,
    fuse_mode: str,
    fuse_percentile: float,
    alpha: float = 0.00014,
    smoothing_sigma: float = 2.0,
) -> np.ndarray:
    total_frames = int(frames_t.shape[0])
    start_min = 0
    start_max = total_frames - window_size
    starts = list(range(start_min, start_max + 1, window_step)) if start_max >= start_min else []
    if not starts:
        s = max(0, total_frames - window_size)
        starts = [s]

    maps = []
    for s in starts:
        e = s + window_size
        bh = compute_bioheat(
            frames_t,
            mask_t,
            start=s,
            end=e,
            smoothing_sigma=smoothing_sigma,
            pixel_size_mm=mm_per_px,
            alpha=alpha,
        )
        maps.append(bh)

    stack = torch.stack(maps, dim=0)  # (n_win, H, W)
    if late_only_fusion and stack.shape[0] >= 2:
        stack = stack[stack.shape[0] // 2 :]

    if fuse_mode == "mean":
        fused = stack.mean(dim=0)
    elif fuse_mode == "percentile":
        fused = torch.quantile(stack, float(fuse_percentile) / 100.0, dim=0)
    else:
        fused = torch.median(stack, dim=0).values

    return fused.detach().cpu().numpy()


def graphcut_binary(
    score_map: np.ndarray,
    mask_bool: np.ndarray,
    lam: float,
    fg_pct: float,
    bg_pct: float,
    gamma: float,
    min_component_px: int,
) -> np.ndarray:
    return segment_graphcut_contrast(
        score_map.astype(np.float32),
        mask_bool.astype(np.uint8),
        lam=float(lam),
        fg_pct=float(fg_pct),
        bg_pct=float(bg_pct),
        gamma=float(gamma),
        min_component_px=int(min_component_px),
    ).astype(np.uint8)


def _masked_vmax(arr: np.ndarray, mask: np.ndarray, pct: float = 99.0) -> float:
    vals = arr[mask]
    if vals.size == 0:
        return float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
    return float(np.percentile(vals, pct))


def _imshow_masked(ax, arr: np.ndarray, mask: np.ndarray, cmap: str, vmin: float, vmax: float, title: str):
    masked = np.where(mask, arr, np.nan)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    return im


def _mamba_label(kind: str) -> str:
    if kind == "artifact":
        return "Mamba Artifact"
    if kind == "thermal":
        return "Mamba Thermal"
    return "Mamba PINN"


def save_comparison_figure(
    save_path: Path,
    mean_frame: np.ndarray,
    mask: np.ndarray,
    bioheat_map: np.ndarray,
    bioheat_bin: np.ndarray,
    mamba_map: np.ndarray,
    mamba_bin: np.ndarray,
    patient: str,
    measurement: str,
    clip_pct: float,
    mamba_label: str,
):
    # Layout (2 rows x 4 cols):
    # Row 1: mean frame | bioheat overlay | bioheat fused | bioheat graphcut (binary)
    # Row 2: mean frame | mamba overlay | mamba map | mamba graphcut (binary)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Common grayscale scaling
    vmin = float(np.percentile(mean_frame, 1))
    vmax = float(np.percentile(mean_frame, 99))

    # (0,0) grayscale mean frame (no mask)
    ax = axes[0, 0]
    im = ax.imshow(mean_frame, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Mean frame (grayscale)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (0,1) Bioheat graphcut overlay on grayscale (orange)
    ax = axes[0, 1]
    ax.imshow(mean_frame, cmap="gray", vmin=vmin, vmax=vmax)
    overlay = np.zeros((*bioheat_bin.shape, 4), dtype=np.float32)
    overlay[..., 1] = 0.6  # green-ish
    overlay[..., 0] = 1.0  # red -> orange
    overlay[..., 3] = (bioheat_bin.astype(bool) & mask).astype(np.float32) * 0.45
    ax.imshow(overlay)
    ax.set_title("Bioheat graphcut overlay")
    ax.axis("off")

    # (0,2) Bioheat fused
    ax = axes[0, 2]
    vmax_bh = _masked_vmax(bioheat_map, mask, clip_pct)
    im = _imshow_masked(ax, bioheat_map, mask, "hot", 0.0, vmax_bh, f"Bioheat fused (clip {clip_pct:.0f}%)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (0,3) Bioheat graphcut
    ax = axes[0, 3]
    ax.imshow(bioheat_bin, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Bioheat graphcut (binary)")
    ax.axis("off")

    # (1,0) grayscale mean frame again
    ax = axes[1, 0]
    im = ax.imshow(mean_frame, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Mean frame (grayscale)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (1,1) Mamba graphcut overlay on grayscale (red)
    ax = axes[1, 1]
    ax.imshow(mean_frame, cmap="gray", vmin=vmin, vmax=vmax)
    overlay = np.zeros((*mamba_bin.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0  # red
    overlay[..., 3] = (mamba_bin.astype(bool) & mask).astype(np.float32) * 0.5
    ax.imshow(overlay)
    ax.set_title(f"{mamba_label} graphcut overlay")
    ax.axis("off")

    # (1,2) Mamba map
    ax = axes[1, 2]
    vmax_m = _masked_vmax(mamba_map, mask, clip_pct)
    im = _imshow_masked(ax, mamba_map, mask, "hot", 0.0, vmax_m, f"{mamba_label} (clip {clip_pct:.0f}%)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (1,3) Mamba graphcut
    ax = axes[1, 3]
    ax.imshow(mamba_bin, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{mamba_label} graphcut (binary)")
    ax.axis("off")

    plt.suptitle(f"{patient}/{measurement} – Bioheat vs {mamba_label} + Graphcut", fontsize=16, fontweight="bold")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(
    mamba_model_path: Path,
    mamba_kind: str,
    h5_dir: Path,
    mask_path: Path,
    output_dir: Path,
    patients: List[str],
    measurements: List[str],
    device: str,
    # bioheat config
    window_size: int,
    window_step: int,
    late_only_fusion: bool,
    fuse_mode: str,
    fuse_percentile: float,
    # mamba config
    mamba_num_frames: int,
    mamba_target_size: Tuple[int, int],
    # graphcut config
    graphcut_lambda: float,
    graphcut_fg_pct: float,
    graphcut_bg_pct: float,
    graphcut_gamma: float,
    graphcut_min_component_px: int,
    # visualization
    clip_percentile: float,
):
    device_t = _to_device(device)
    mamba_model = load_mamba_model(str(mamba_model_path), mamba_kind, device=str(device_t))
    mamba_label = _mamba_label(mamba_kind)

    for patient in patients:
        for meas in measurements:
            pid = f"{patient}{meas}"
            frames_np = load_frames_h5(h5_dir, pid)
            if frames_np is None:
                print(f"[skip] {pid}: frames not found")
                continue

            _, H, W = frames_np.shape
            focus_mm = load_focus(h5_dir, patient) or 50.0
            mm_per_px = get_mm_per_pixel(focus_mm, W, H)
            mask_np = load_mask(pid, (H, W), mask_path).astype(bool)

            frames_t = torch.from_numpy(frames_np.astype(np.float32)).to(device_t)
            mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(device_t)

            bioheat_map = compute_bioheat_fused(
                frames_t=frames_t,
                mask_t=mask_t,
                mm_per_px=mm_per_px,
                window_size=window_size,
                window_step=window_step,
                late_only_fusion=late_only_fusion,
                fuse_mode=fuse_mode,
                fuse_percentile=fuse_percentile,
            )

            mamba_map_small = run_mamba_inference(
                model=mamba_model,
                frames=frames_np,
                device=str(device_t),
                num_frames=mamba_num_frames,
                target_size=mamba_target_size,
            )
            mamba_map = resize_to_shape(mamba_map_small, (H, W))

            bioheat_bin = graphcut_binary(
                score_map=bioheat_map,
                mask_bool=mask_np,
                lam=graphcut_lambda,
                fg_pct=graphcut_fg_pct,
                bg_pct=graphcut_bg_pct,
                gamma=graphcut_gamma,
                min_component_px=graphcut_min_component_px,
            )
            mamba_bin = graphcut_binary(
                score_map=mamba_map,
                mask_bool=mask_np,
                lam=graphcut_lambda,
                fg_pct=graphcut_fg_pct,
                bg_pct=graphcut_bg_pct,
                gamma=graphcut_gamma,
                min_component_px=graphcut_min_component_px,
            )

            mean_frame = frames_np.mean(axis=0)
            save_path = output_dir / f"compare_bioheat_mamba_graphcut_{patient}_{meas}.png"
            save_comparison_figure(
                save_path=save_path,
                mean_frame=mean_frame,
                mask=mask_np,
                bioheat_map=bioheat_map,
                bioheat_bin=bioheat_bin,
                mamba_map=mamba_map,
                mamba_bin=mamba_bin,
                patient=patient,
                measurement=meas,
                clip_pct=clip_percentile,
                mamba_label=mamba_label,
            )

            print(f"[ok] {pid}: saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Bioheat and Mamba outputs with graphcut segmentation.")
    parser.add_argument("--mamba_model", type=str, required=True, help="Path to Mamba checkpoint (.pth).")
    parser.add_argument(
        "--mamba_kind",
        type=str,
        choices=["pinn"],
        default="pinn",
        help="Which Mamba architecture to instantiate.",
    )
    parser.add_argument("--h5_dir", type=str, default="/dodrio/scratch/projects/starting_2025_090/H5")
    parser.add_argument(
        "--mask_path",
        type=str,
        default="/dodrio/scratch/projects/starting_2025_090/Annotations/coco_annotations_cooled_area.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/dodrio/scratch/projects/starting_2025_090/output_n2n/e2e_bioheat_graphcut_comparisons",
    )
    parser.add_argument("--patients", type=str, nargs="+", default=[f"P{i:02d}" for i in range(1, 26)])
    parser.add_argument("--measurements", type=str, nargs="+", default=["M01", "M02", "M03", "M04"])
    parser.add_argument("--device", type=str, default="cuda")

    # Bioheat fusion settings
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--window_step", type=int, default=50)
    parser.add_argument("--late_only_fusion", action="store_true")
    parser.add_argument("--fuse_mode", type=str, choices=["median", "mean", "percentile"], default="median")
    parser.add_argument("--fuse_percentile", type=float, default=30.0)

    # Mamba input sizing
    parser.add_argument(
        "--mamba_num_frames",
        type=int,
        default=None,
        help="Number of frames fed to the Mamba model. Defaults: pinn=480, thermal/artifact=200.",
    )
    parser.add_argument("--mamba_target_size", type=int, nargs=2, default=(240, 320))

    # Graphcut settings
    parser.add_argument("--graphcut_lambda", type=float, default=10.0)
    parser.add_argument("--graphcut_fg_pct", type=float, default=85.0)
    parser.add_argument("--graphcut_bg_pct", type=float, default=30.0)
    parser.add_argument("--graphcut_gamma", type=float, default=1.5)
    parser.add_argument("--graphcut_min_component_px", type=int, default=50)

    # Visualization
    parser.add_argument("--clip_percentile", type=float, default=99.0)

    args = parser.parse_args()

    if args.mamba_num_frames is None:
        args.mamba_num_frames = 480 if args.mamba_kind == "pinn" else 200

    run(
        mamba_model_path=Path(args.mamba_model),
        mamba_kind=args.mamba_kind,
        h5_dir=Path(args.h5_dir),
        mask_path=Path(args.mask_path),
        output_dir=Path(args.output_dir),
        patients=args.patients,
        measurements=args.measurements,
        device=args.device,
        window_size=args.window_size,
        window_step=args.window_step,
        late_only_fusion=args.late_only_fusion,
        fuse_mode=args.fuse_mode,
        fuse_percentile=args.fuse_percentile,
        mamba_num_frames=int(args.mamba_num_frames),
        mamba_target_size=tuple(args.mamba_target_size),
        graphcut_lambda=args.graphcut_lambda,
        graphcut_fg_pct=args.graphcut_fg_pct,
        graphcut_bg_pct=args.graphcut_bg_pct,
        graphcut_gamma=args.graphcut_gamma,
        graphcut_min_component_px=args.graphcut_min_component_px,
        clip_percentile=args.clip_percentile,
    )


if __name__ == "__main__":
    main()
