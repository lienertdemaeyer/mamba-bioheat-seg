"""
Graph-cut segmentation with edge-aware contrast weighting.

This module provides a graph-cut based segmentation method that uses
intensity contrast to define edge weights, producing smooth boundaries
that respect image gradients.
"""

import numpy as np
import cv2

try:
    import maxflow
except ImportError:
    maxflow = None


def _add_edges_bulk(g, u, v, w):
    """Add undirected edges in bulk if possible, else per-edge."""
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()
    w = np.asarray(w).ravel()
    if u.size == 0:
        return
    if hasattr(g, "add_edges"):
        g.add_edges(u, v, w, w)
    else:
        for ui, vi, wi in zip(u, v, w):
            g.add_edge(int(ui), int(vi), float(wi), float(wi))


def _auto_beta_from_masked_diffs(I: np.ndarray, mask_bool: np.ndarray) -> float:
    """Robust beta estimation from neighbor differences inside mask."""
    dh = (I[:, 1:] - I[:, :-1]) ** 2
    dv = (I[1:, :] - I[:-1, :]) ** 2
    mh = (mask_bool[:, 1:] & mask_bool[:, :-1])
    mv = (mask_bool[1:, :] & mask_bool[:-1, :])
    d = np.concatenate([dh[mh].ravel(), dv[mv].ravel()])
    med = float(np.median(d)) if d.size else 1.0
    return 1.0 / (2.0 * (med + 1e-12))


def _filter_components_area_energy(
    binary: np.ndarray,
    score_map: np.ndarray,
    min_area: int = 0,
    min_sum: float = 0.0,
    min_mean: float = 0.0,
) -> np.ndarray:
    """Keep components by area/sum/mean criteria."""
    b = (binary > 0).astype(np.uint8)
    if not b.any():
        return b
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    if n <= 1:
        return b
    lab = labels.ravel()
    sc = score_map.astype(np.float32).ravel()
    area = np.bincount(lab)
    ssum = np.bincount(lab, weights=sc)
    mean = ssum / np.maximum(area, 1)
    keep = np.ones(n, dtype=bool)
    keep[0] = False
    if min_area > 0:
        keep &= (area >= min_area)
    if min_sum > 0:
        keep &= (ssum >= float(min_sum))
    if min_mean > 0:
        keep &= (mean >= float(min_mean))
    return keep[labels].astype(np.uint8)


def segment_graphcut_contrast(
    seg_map_np: np.ndarray,
    mask_np: np.ndarray,
    lam: float = 10.0,
    fg_pct: float = 85.0,
    bg_pct: float = 30.0,
    gamma: float = 1.5,
    min_component_px: int = 50,
    unary_blur_sigma: float = 0.4,
    post_smooth_radius: int = 1,
    beta: float | None = None,
    use_diagonals: bool = True,
    p_fore_median_ksize: int = 3,
    energy_score_sigma: float = 10.0,
    min_component_sum: float = 0.0,
    min_component_mean: float = 0.0,
) -> np.ndarray:
    """
    Segment an image using graph-cut with edge-aware contrast weighting.

    Parameters
    ----------
    seg_map_np : np.ndarray
        2D input image (e.g., segmentation score map or fused bioheat map).
    mask_np : np.ndarray
        2D mask indicating valid region for segmentation.
    lam : float
        Smoothness weight for pairwise terms. Default 10.0.
    fg_pct : float
        Percentile threshold for foreground seeds (0-100). Default 85.0.
    bg_pct : float
        Percentile threshold for background seeds (0-100). Default 30.0.
    gamma : float
        Power exponent for foreground probability. Default 1.5.
    min_component_px : int
        Minimum component area in pixels. Default 50.
    unary_blur_sigma : float, optional
        Sigma for Gaussian blur on unary terms. Default 0.4.
    post_smooth_radius : int, optional
        Radius for morphological closing post-processing. Default 1.
    beta : float or None, optional
        Edge weight decay parameter. If None, auto-estimated from image.
    use_diagonals : bool, optional
        Whether to include diagonal edges. Default True.
    p_fore_median_ksize : int, optional
        Kernel size for median filter on foreground probability. Default 3.
    energy_score_sigma : float, optional
        Sigma for energy score computation. Default 10.0.
    min_component_sum : float, optional
        Minimum sum of score values in a component. Default 0.0.
    min_component_mean : float, optional
        Minimum mean score value in a component. Default 0.0.

    Returns
    -------
    np.ndarray
        Binary segmentation mask (uint8).
    
    Raises
    ------
    ImportError
        If PyMaxflow is not installed.
    
    Example
    -------
    >>> from bioheat_fusion_gpu.graphcut import segment_graphcut_contrast
    >>> binary_mask = segment_graphcut_contrast(
    ...     fused_map, 
    ...     roi_mask,
    ...     lam=10.0,
    ...     fg_pct=85.0,
    ...     bg_pct=30.0,
    ... )
    """
    if maxflow is None:
        raise ImportError("PyMaxflow is required; install with `pip install PyMaxflow`.")

    mask_bool = mask_np.astype(bool)
    img = seg_map_np.astype(np.float32)

    # Light unary smoothing only (don't overdo it)
    if unary_blur_sigma and unary_blur_sigma > 0:
        k = int(6 * unary_blur_sigma + 1) | 1
        img_u = cv2.GaussianBlur(img, (k, k), unary_blur_sigma)
    else:
        img_u = img.copy()
    img_u[~mask_bool] = 0.0

    valid_vals = img_u[mask_bool & (img_u > 0)]
    if valid_vals.size == 0:
        return np.zeros_like(img_u, dtype=np.uint8)

    fg_thr = float(np.percentile(valid_vals, fg_pct))
    bg_thr = float(np.percentile(valid_vals, bg_pct))
    if fg_thr <= bg_thr:
        fg_thr = bg_thr + 1e-6

    norm = (img_u - bg_thr) / (fg_thr - bg_thr)
    norm = np.clip(norm, 0.0, 1.0)
    p_fore = np.power(norm, gamma).astype(np.float32)
    if p_fore_median_ksize and p_fore_median_ksize >= 3:
        k = int(p_fore_median_ksize) | 1
        p8 = np.clip(p_fore * 255.0, 0, 255).astype(np.uint8)
        p8 = cv2.medianBlur(p8, k)
        p_fore = p8.astype(np.float32) / 255.0
    p_fore = np.clip(p_fore, 1e-4, 1.0 - 1e-4)
    p_fore[~mask_bool] = 1e-4
    p_back = 1.0 - p_fore

    background_cost = (-np.log(p_back)).astype(np.float32)
    foreground_cost = (-np.log(p_fore)).astype(np.float32)

    # Hard seeds
    fg_seed = (img_u >= fg_thr) & mask_bool
    bg_seed = (img_u <= bg_thr) & mask_bool
    background_cost[fg_seed] = 1e6
    foreground_cost[fg_seed] = 0.0
    background_cost[bg_seed] = 0.0
    foreground_cost[bg_seed] = 1e6

    background_cost[~mask_bool] = 0.0
    foreground_cost[~mask_bool] = 1e6

    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes(img_u.shape)
    g.add_grid_tedges(nodes, background_cost, foreground_cost)

    # ---- Edge-aware pairwise weights computed from the *original* (or lightly smoothed) image ----
    I = img.astype(np.float32)

    if beta is None:
        beta = _auto_beta_from_masked_diffs(I, mask_bool)

    # horizontal edges
    w_h = lam * np.exp(-beta * (I[:, 1:] - I[:, :-1]) ** 2)
    w_h *= (mask_bool[:, 1:] & mask_bool[:, :-1])

    u = nodes[:, :-1].ravel()
    v = nodes[:,  1:].ravel()
    _add_edges_bulk(g, u, v, w_h.ravel().astype(np.float32))

    # vertical edges
    w_v = lam * np.exp(-beta * (I[1:, :] - I[:-1, :]) ** 2)
    w_v *= (mask_bool[1:, :] & mask_bool[:-1, :])

    u = nodes[:-1, :].ravel()
    v = nodes[ 1:, :].ravel()
    _add_edges_bulk(g, u, v, w_v.ravel().astype(np.float32))

    # diagonals (optional) -> reduces "Manhattan corners" but preserves edges due to contrast weights
    if use_diagonals:
        lam_d = lam / np.sqrt(2.0)

        w_d1 = lam_d * np.exp(-beta * (I[1:, 1:] - I[:-1, :-1]) ** 2)
        w_d1 *= (mask_bool[1:, 1:] & mask_bool[:-1, :-1])
        u = nodes[:-1, :-1].ravel()
        v = nodes[ 1:,  1:].ravel()
        _add_edges_bulk(g, u, v, w_d1.ravel().astype(np.float32))

        w_d2 = lam_d * np.exp(-beta * (I[1:, :-1] - I[:-1, 1:]) ** 2)
        w_d2 *= (mask_bool[1:, :-1] & mask_bool[:-1, 1:])
        u = nodes[:-1,  1:].ravel()
        v = nodes[ 1:, :-1].ravel()
        _add_edges_bulk(g, u, v, w_d2.ravel().astype(np.float32))

    g.maxflow()
    segments = g.get_grid_segments(nodes)
    binary = (~segments).astype(np.uint8)
    binary[~mask_bool] = 0

    if min_component_px > 0:
        binary = _filter_components_area_energy(binary, np.ones_like(I, dtype=np.float32), min_area=int(min_component_px))
        binary[~mask_bool] = 0

    if (min_component_sum and min_component_sum > 0) or (min_component_mean and min_component_mean > 0):
        score_blur = cv2.GaussianBlur(I, (0, 0), float(energy_score_sigma))
        score_map = np.maximum(I - score_blur, 0.0)
        score_map[~mask_bool] = 0.0
        binary = _filter_components_area_energy(
            binary=binary,
            score_map=score_map,
            min_area=0,
            min_sum=float(min_component_sum),
            min_mean=float(min_component_mean),
        )
        binary[~mask_bool] = 0

    if post_smooth_radius and post_smooth_radius > 0:
        r = int(post_smooth_radius)
        k = 2 * r + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ker)
        binary[~mask_bool] = 0

    return binary


