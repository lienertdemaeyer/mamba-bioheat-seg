from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _require(cfg: DictConfig, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if not hasattr(cur, part):
            raise KeyError(f"Missing config key: {dotted}")
        cur = getattr(cur, part)
    return cur


def _task_name(cfg: DictConfig) -> str:
    task = _require(cfg, "task.name")
    if not isinstance(task, str) or not task:
        raise ValueError("`task.name` must be a non-empty string")
    return task


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(str(p))


def _coerce_str_list(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        # allow comma and/or whitespace separators
        parts = [p.strip() for p in s.replace(",", " ").split()]
        return [p for p in parts if p]
    return [str(x) for x in list(val)]


def _run_train_mamba_pinn(cfg: DictConfig) -> None:
    from .train_mamba_physics_informed import run_training

    run_training(cfg)


def _run_train_true_e2e(cfg: DictConfig) -> None:
    from .train_true_e2e import run_training

    run_training(cfg)


def _run_infer_bioheat_vs_mamba_graphcut(cfg: DictConfig) -> None:
    from .compare_bioheat_mamba_graphcut import run as run_compare

    device = str(_require(cfg, "inference.device"))
    mamba_model = str(_require(cfg, "inference.mamba.model"))
    mamba_kind = str(_require(cfg, "inference.mamba.kind"))

    h5_dir = _as_path(_require(cfg, "paths.h5_dir"))
    mask_path = _as_path(_require(cfg, "paths.mask_path"))
    output_dir = _as_path(_require(cfg, "inference.output_dir"))

    patients = _coerce_str_list(_require(cfg, "inference.patients"))
    measurements = _coerce_str_list(_require(cfg, "inference.measurements"))

    # Bioheat fusion
    window_size = int(_require(cfg, "bioheat.window_size"))
    window_step = int(_require(cfg, "bioheat.window_step"))
    late_only_fusion = bool(_require(cfg, "bioheat.late_only_fusion"))
    fuse_mode = str(_require(cfg, "bioheat.fuse_mode"))
    fuse_percentile = float(_require(cfg, "bioheat.fuse_percentile"))

    # Mamba input sizing
    mamba_num_frames = int(_require(cfg, "inference.mamba.num_frames"))
    mamba_target_h = int(_require(cfg, "inference.mamba.target_h"))
    mamba_target_w = int(_require(cfg, "inference.mamba.target_w"))

    # Graphcut params
    graphcut_lambda = float(_require(cfg, "graphcut.lambda"))
    graphcut_fg_pct = float(_require(cfg, "graphcut.fg_pct"))
    graphcut_bg_pct = float(_require(cfg, "graphcut.bg_pct"))
    graphcut_gamma = float(_require(cfg, "graphcut.gamma"))
    graphcut_min_component_px = int(_require(cfg, "graphcut.min_component_px"))

    clip_percentile = float(_require(cfg, "viz.clip_percentile"))

    run_compare(
        mamba_model_path=Path(mamba_model),
        mamba_kind=mamba_kind,
        h5_dir=h5_dir,
        mask_path=mask_path,
        output_dir=output_dir,
        patients=patients,
        measurements=measurements,
        device=device,
        window_size=window_size,
        window_step=window_step,
        late_only_fusion=late_only_fusion,
        fuse_mode=fuse_mode,
        fuse_percentile=fuse_percentile,
        mamba_num_frames=mamba_num_frames,
        mamba_target_size=(mamba_target_h, mamba_target_w),
        graphcut_lambda=graphcut_lambda,
        graphcut_fg_pct=graphcut_fg_pct,
        graphcut_bg_pct=graphcut_bg_pct,
        graphcut_gamma=graphcut_gamma,
        graphcut_min_component_px=graphcut_min_component_px,
        clip_percentile=clip_percentile,
    )


_TASKS = {
    "train_mamba_pinn": _run_train_mamba_pinn,
    "train_true_e2e": _run_train_true_e2e,
    "infer_bioheat_vs_mamba_graphcut": _run_infer_bioheat_vs_mamba_graphcut,
}


@hydra.main(version_base=None, config_path="conf", config_name="app")
def main(cfg: DictConfig) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    task = _task_name(cfg)
    fn = _TASKS.get(task)
    if fn is None:
        raise ValueError(f"Unknown task '{task}'. Available: {', '.join(sorted(_TASKS))}")

    log.info("Hydra app started")
    log.info("Working directory: %s", os.getcwd())
    log.info("Task: %s", task)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    fn(cfg)


if __name__ == "__main__":
    main()
