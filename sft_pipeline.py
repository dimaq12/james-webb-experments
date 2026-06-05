"""
Universal sft_torch spectral analysis pipeline.
Accepts any image (PNG, JPG, FITS), runs full operator analysis,
produces metrics JSON + side-by-side reveal figures with iterated residuals.

Usage:
    python sft_pipeline.py image.jpg
    python sft_pipeline.py image.fits --residual-depth 3 --stride 80
    python sft_pipeline.py image.png --max-regions 200 --output-dir results/
"""

import sys, os, json, time, argparse, warnings
import numpy as np
import torch, sft_torch as sft

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_N_GRID = 8
DEFAULT_M_PARAMS = 4
DEFAULT_REGION_SIZE = 64
DEFAULT_STRIDE = 64
DEFAULT_MAX_REGIONS = 200
DEFAULT_RESIDUAL_DEPTH = 2
IMAGE_MAX_DIM = 2000  # auto-downscale if larger


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_image(path):
    """Load PNG/JPG/FITS → float64 grayscale array [0, 1]."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fits", ".fit"):
        from astropy.io import fits
        hdul = fits.open(path)
        # find science extension
        for ext_idx in range(len(hdul)):
            d = hdul[ext_idx].data
            if d is not None and hasattr(d, "shape") and len(d.shape) >= 2:
                break
        else:
            raise ValueError("No 2D data in FITS")
        arr = np.array(d, dtype=np.float64, copy=True)
        arr[~np.isfinite(arr)] = 0.0
        if arr.max() > 1.0:
            arr = arr / np.percentile(arr[arr > 0], 99.9) if np.any(arr > 0) else arr
            arr = np.clip(arr, 0, 1)
    else:
        from PIL import Image
        img = Image.open(path).convert("L")
        # downscale if too large
        if max(img.size) > IMAGE_MAX_DIM:
            ratio = IMAGE_MAX_DIM / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        arr = np.array(img, dtype=np.float64) / 255.0
    return arr


# ─────────────────────────────────────────────
# OPERATOR SETUP
# ─────────────────────────────────────────────
def build_basis(n_grid, m_params):
    N = n_grid * n_grid
    basis = []
    for d in range(m_params):
        B = np.zeros((N, N))
        for i in range(N):
            xi, yi = i % n_grid, i // n_grid
            j = yi * n_grid + ((xi + d) % n_grid)
            B[i, j] = 1.0
            B[j, i] = 1.0
        basis.append(torch.from_numpy(B))
    return torch.stack(basis).to(torch.float64)


# ─────────────────────────────────────────────
# SINGLE-BLOCK ANALYSIS
# ─────────────────────────────────────────────
def analyze_block(block, basis, n_grid, use_perturbation=False):
    N = n_grid * n_grid
    px = block[::block.shape[0] // n_grid, ::block.shape[1] // n_grid][:n_grid, :n_grid].ravel()
    px = px - np.min(px) + 0.1
    L = np.zeros((N, N))
    for i in range(N):
        xi, yi = i % n_grid, i // n_grid
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = xi + dx, yi + dy
            if 0 <= nx < n_grid and 0 <= ny < n_grid:
                j = ny * n_grid + nx
                L[i, j] -= 1
                L[i, i] += 1
    A0 = torch.diag(torch.from_numpy(px)) + torch.from_numpy(L) * 0.3
    prog = sft.operator(A0.to(torch.float64), basis).compile("cpu")
    fam = prog.family
    tail = prog.tail(strict=False)
    kap = prog.kappa(method="perturbation", n_pert=3, eps=0.1) if use_perturbation else prog.kappa(method="hessian")

    kw = kap.summary()["value"]
    alpha = float(tail.alpha if tail.alpha else 0)
    om = tail.omega.cpu().numpy()
    om_norm = float(np.linalg.norm(om))
    lam = fam.lam0
    gaps = torch.diff(lam)
    dec = prog.decompose()
    sv = dec.singular

    return {
        "kw": float(kw),
        "alpha": alpha,
        "omega_vec": om.tolist(),
        "omega_norm": om_norm,
        "lam0_min": float(lam[0]),
        "lam0_max": float(lam[-1]),
        "lam_std": float(lam.std()),
        "lam_range": float(lam[-1] - lam[0]),
        "gap_median": float(gaps.median()),
        "gap_min": float(gaps.min()),
        "gap_max": float(gaps.max()),
        "rank": fam.W_rank,
        "cond": float(fam.condition_number()),
        "sv_max": float(sv[0]),
        "sv_min": float(sv[-1]) if len(sv) > 0 else 0,
        "sv_std": float(sv.std()) if len(sv) > 0 else 0,
        "complexity": float(fam.complexity),
        "pixel_mean": float(block.mean()),
        "pixel_std": float(block.std()),
    }


# ─────────────────────────────────────────────
# SCAN IMAGE
# ─────────────────────────────────────────────
def scan_image(arr, basis, n_grid, region_size, stride, max_regions, use_perturbation=False):
    H, W = arr.shape
    stats = []
    for i in range(0, H - region_size, stride):
        if len(stats) >= max_regions:
            break
        for j in range(0, W - region_size, stride):
            if len(stats) >= max_regions:
                break
            block = arr[i:i + region_size, j:j + region_size]
            if block.shape != (region_size, region_size):
                continue
            if block.max() - block.min() < 0.001:
                continue
            s = analyze_block(block, basis, n_grid, use_perturbation)
            s["i"] = i
            s["j"] = j
            s["ci"] = i + region_size // 2
            s["cj"] = j + region_size // 2
            stats.append(s)
    return stats


# ─────────────────────────────────────────────
# ITERATED REGRESSION
# ─────────────────────────────────────────────
def regress(y, X):
    try:
        c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        c = np.linalg.pinv(X) @ y
    pred = X @ c
    resid = y - pred
    R2 = 1 - np.var(resid) / np.var(y) if np.var(y) > 1e-20 else 0.0
    return pred, resid, float(R2)


def iterated_regression(stats, depth=2):
    kw = np.array([s["kw"] for s in stats])
    alpha = np.array([s["alpha"] for s in stats])
    lam_std = np.array([s["lam_std"] for s in stats])
    lam_range = np.array([s["lam_range"] for s in stats])
    gap_median = np.array([s["gap_median"] for s in stats])
    gap_min = np.array([s["gap_min"] for s in stats])
    sv_std = np.array([s["sv_std"] for s in stats])
    cond = np.array([s["cond"] for s in stats])
    omega_norm = np.array([s["omega_norm"] for s in stats])
    pm = np.array([s["pixel_mean"] for s in stats])
    ps = np.array([s["pixel_std"] for s in stats])
    ci_arr = np.array([s["ci"] for s in stats])
    cj_arr = np.array([s["cj"] for s in stats])
    ci_std = ci_arr.std() if ci_arr.std() > 1 else 1.0
    cj_std = cj_arr.std() if cj_arr.std() > 1 else 1.0
    gin = (ci_arr - ci_arr.mean()) / ci_std
    gjn = (cj_arr - cj_arr.mean()) / cj_std

    lam0_min = np.array([s["lam0_min"] for s in stats])
    lam0_max = np.array([s["lam0_max"] for s in stats])
    gap_max = np.array([s["gap_max"] for s in stats])
    sv_max = np.array([s["sv_max"] for s in stats])
    sv_min = np.array([s["sv_min"] for s in stats])
    complexity = np.array([s["complexity"] for s in stats])

    levels = []
    residuals = [kw.copy()]  # residuals[0] = kw, residuals[1] = resid0, etc.
    cum_r2 = 0.0

    # L0: brightness
    X0 = np.column_stack([pm, ps, np.ones_like(pm)])
    _, resid0, r2_0 = regress(kw, X0)
    levels.append({"name": "L0: κ_W ~ яркость+контраст", "R2": r2_0, "features": ["pixel_mean", "pixel_std"]})
    residuals.append(resid0)
    cum_r2 += r2_0

    if depth >= 1:
        X1 = np.column_stack([
            alpha, omega_norm, lam_std, lam_range, gap_median, gap_min, sv_std, cond,
            gin, gjn, gin**2, gjn**2, gin * gjn, np.ones_like(alpha),
        ])
        _, resid1, r2_1 = regress(residuals[-1], X1)
        delta1 = (1 - cum_r2) * r2_1
        cum_r2 += delta1
        levels.append({"name": "L1: resid0 ~ α+Ω+спектр+позиция", "R2": r2_1, "delta": delta1,
                       "features": ["alpha", "omega_norm", "lam_*", "gap_*", "sv_*", "cond", "position"]})
        residuals.append(resid1)

    if depth >= 2:
        X2 = np.column_stack([
            lam0_min, lam0_max, gap_max, sv_max, sv_min, complexity, np.ones_like(lam0_min),
        ])
        _, resid2, r2_2 = regress(residuals[-1], X2)
        delta2 = (1 - cum_r2) * r2_2
        cum_r2 += delta2
        levels.append({"name": "L2: resid1 ~ границы спектра+gap", "R2": r2_2, "delta": delta2,
                       "features": ["lam0_min", "lam0_max", "gap_max", "sv_max", "sv_min", "complexity"]})
        residuals.append(resid2)

    if depth >= 3:
        # L3: spatial autocorrelation + cross terms
        X3 = np.column_stack([
            gin * alpha, gjn * alpha, gin * omega_norm, gjn * omega_norm,
            gin**3, gjn**3, np.ones_like(alpha),
        ])
        _, resid3, r2_3 = regress(residuals[-1], X3)
        delta3 = (1 - cum_r2) * r2_3
        cum_r2 += delta3
        levels.append({"name": "L3: resid2 ~ перекрёстные α·pos", "R2": r2_3, "delta": delta3,
                       "features": ["gin*alpha", "gjn*alpha", "gin*om", "gin^3", "gjn^3"]})
        residuals.append(resid3)

    return {
        "levels": levels,
        "cumulative_R2": float(cum_r2),
        "residuals": [r.tolist() for r in residuals],
        "final_residual_std": float(residuals[-1].std()),
        "kappa_std": float(kw.std()),
        "compression_ratio": float(residuals[-1].std() / kw.std()),
    }


# ─────────────────────────────────────────────
# BUILD DENSE MAP
# ─────────────────────────────────────────────
def build_dense_map(vals, ci, cj, H, W, sigma=2.5):
    max_dim = 1200
    scale = min(1.0, max_dim / max(H, W))
    h_s = max(int(H * scale), 10)
    w_s = max(int(W * scale), 10)
    gy, gx = np.mgrid[0:h_s, 0:w_s]
    ci_s = ci * scale
    cj_s = cj * scale
    try:
        dense = griddata((ci_s, cj_s), vals, (gy, gx), method="linear", fill_value=np.nan)
    except Exception:
        dense = griddata((ci_s, cj_s), vals, (gy, gx), method="nearest", fill_value=np.nan)
    mask = ~np.isnan(dense)
    dense[~mask] = 0
    if mask.sum() > 0:
        dense = gaussian_filter(dense, sigma=max(0.5, sigma * scale))
    dense[~mask] = np.nan
    return dense


# ─────────────────────────────────────────────
# FIGURE GENERATION
# ─────────────────────────────────────────────
def generate_figure(arr, stats, reg_results, outpath, residual_depth=2):
    kw = np.array([s["kw"] for s in stats])
    alpha = np.array([s["alpha"] for s in stats])
    om_norm = np.array([s["omega_norm"] for s in stats])
    ci_arr = np.array([s["ci"] for s in stats])
    cj_arr = np.array([s["cj"] for s in stats])

    residuals = reg_results["residuals"]
    H, W = arr.shape

    kw_map = build_dense_map(kw, ci_arr, cj_arr, H, W)
    om_map = build_dense_map(om_norm, ci_arr, cj_arr, H, W)

    # residual maps
    resid_maps = []
    for ridx in range(1, min(len(residuals), residual_depth + 2)):
        r_vals = np.array(residuals[ridx])
        resid_maps.append(build_dense_map(r_vals, ci_arr, cj_arr, H, W))

    # Downscale background for display
    max_bg = 1200
    if max(H, W) > max_bg:
        from PIL import Image
        scale_bg = max_bg / max(H, W)
        h_bg = int(H * scale_bg); w_bg = int(W * scale_bg)
        bg = np.array(Image.fromarray((arr * 255).astype(np.uint8)).resize((w_bg, h_bg), Image.LANCZOS), dtype=np.float64) / 255.0
    else:
        bg = arr.copy()
    bg[~np.isfinite(bg)] = 0
    h_disp, w_disp = bg.shape

    n_cols = 2 + len(resid_maps)
    n_cols = min(n_cols, 5)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols + 2, 6))
    if n_cols == 1: axes = [axes]

    ax = axes[0]
    ax.imshow(bg, cmap="gray" if bg.max() <= 1.5 else "inferno", origin="lower")
    ax.set_title("Original", fontsize=11, fontweight="bold")

    ax = axes[1]
    ax.imshow(bg, cmap="gray", origin="lower", alpha=0.25)
    im = ax.imshow(kw_map, cmap="plasma", origin="lower", alpha=0.8, interpolation="bilinear")
    ax.set_title(f"κ_W  [{kw.min():.0f}–{kw.max():.0f}]", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    for idx, rmap in enumerate(resid_maps):
        if idx + 2 >= n_cols: break
        ax = axes[idx + 2]
        ax.imshow(bg, cmap="gray", origin="lower", alpha=0.4)
        r = np.array(residuals[idx + 1])
        vlim = max(abs(r.min()), abs(r.max())) * 0.8
        im = ax.imshow(rmap, cmap="RdBu_r", origin="lower", alpha=0.7,
                       interpolation="bilinear", vmin=-vlim, vmax=vlim)
        if idx < len(reg_results["levels"]):
            lvl = reg_results["levels"][idx]
            ax.set_title(f"Resid {idx}  R²={lvl['R2']:.3f}",
                         fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"sft_torch — cum R²={reg_results['cumulative_R2']:.3f}  "
                 f"σ: {kw.std():.0f}→{reg_results['final_residual_std']:.1f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ─────────────────────────────────────────────
# METRICS EXPORT
# ─────────────────────────────────────────────
def build_metrics(stats, reg_results):
    kw = np.array([s["kw"] for s in stats])
    alpha = np.array([s["alpha"] for s in stats])
    om_norm = np.array([s["omega_norm"] for s in stats])

    return {
        "n_regions": len(stats),
        "kappa_W": {
            "min": float(kw.min()), "max": float(kw.max()),
            "mean": float(kw.mean()), "std": float(kw.std()),
            "cv": float(kw.std() / kw.mean()) if kw.mean() > 0 else 0,
        },
        "alpha": {
            "min": float(alpha.min()), "max": float(alpha.max()),
            "mean": float(alpha.mean()), "std": float(alpha.std()),
            "cv": float(alpha.std() / alpha.mean()) if alpha.mean() > 0 else 0,
        },
        "omega_norm": {
            "min": float(om_norm.min()), "max": float(om_norm.max()),
            "mean": float(om_norm.mean()), "std": float(om_norm.std()),
            "cv": float(om_norm.std() / om_norm.mean()) if om_norm.mean() > 0 else 0,
        },
        "mean_rank": float(np.mean([s["rank"] for s in stats])),
        "mean_condition": float(np.mean([s["cond"] for s in stats])),
        "mean_complexity": float(np.mean([s["complexity"] for s in stats])),
        "regression": {lvl["name"]: {"R2": lvl["R2"], "delta": lvl.get("delta", lvl["R2"])}
                       for lvl in reg_results["levels"]},
        "cumulative_R2": reg_results["cumulative_R2"],
        "compression_ratio": reg_results["compression_ratio"],
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Universal sft_torch spectral analysis pipeline")
    parser.add_argument("image", help="Path to image (PNG, JPG, FITS)")
    parser.add_argument("--output-dir", "-o", default="pipeline_out",
                        help="Output directory (default: pipeline_out)")
    parser.add_argument("--residual-depth", "-d", type=int, default=DEFAULT_RESIDUAL_DEPTH,
                        help="Iterated regression depth 0-3 (default: 2)")
    parser.add_argument("--max-regions", "-n", type=int, default=DEFAULT_MAX_REGIONS,
                        help="Max regions to scan (default: 300)")
    parser.add_argument("--stride", "-s", type=int, default=DEFAULT_STRIDE,
                        help="Scan stride in pixels (default: 80)")
    parser.add_argument("--region-size", "-r", type=int, default=DEFAULT_REGION_SIZE,
                        help="Block size in pixels (default: 80)")
    parser.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID,
                        help="Operator grid size (default: 10)")
    parser.add_argument("--perturbation", action="store_true",
                        help="Use perturbation kappa (slower, more accurate)")
    parser.add_argument("--m-params", type=int, default=DEFAULT_M_PARAMS,
                        help="Basis parameters M (default: 5)")
    args = parser.parse_args()

    if args.residual_depth > 3:
        print("Max residual depth is 3")
        args.residual_depth = 3

    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.image))[0]

    # 1. Load
    print(f"[1/5] Loading {args.image} ...")
    arr = load_image(args.image)
    print(f"  shape: {arr.shape[1]}x{arr.shape[0]}  range: [{arr.min():.3f}, {arr.max():.3f}]")

    # 2. Setup
    print(f"[2/5] Building basis (N={args.n_grid}^2={args.n_grid**2}, M={args.m_params}) ...")
    basis = build_basis(args.n_grid, args.m_params)

    # 3. Scan
    print(f"[3/5] Scanning (stride={args.stride}, region={args.region_size}) ...")
    t0 = time.perf_counter()
    stats = scan_image(arr, basis, args.n_grid, args.region_size, args.stride, args.max_regions, args.perturbation)
    dt = time.perf_counter() - t0
    print(f"  {len(stats)} regions in {dt:.0f}s")

    if len(stats) < 5:
        print("ERROR: too few valid regions. Try reducing --stride or --region-size.")
        sys.exit(1)

    # 4. Regression
    print(f"[4/5] Iterated regression (depth={args.residual_depth}) ...")
    reg_results = iterated_regression(stats, depth=args.residual_depth)

    for lvl in reg_results["levels"]:
        print(f"  {lvl['name']}: R² = {lvl['R2']:.4f}")
    print(f"  Cumulative R² = {reg_results['cumulative_R2']:.4f}")
    print(f"  Compression: {reg_results['final_residual_std']:.1f} / {reg_results['kappa_std']:.1f} "
          f"= {reg_results['compression_ratio']:.3f}")

    # 5. Output
    print(f"[5/5] Generating outputs ...")
    metrics = build_metrics(stats, reg_results)

    metrics_path = os.path.join(args.output_dir, f"{basename}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"  {metrics_path}")

    fig_path = os.path.join(args.output_dir, f"{basename}_reveal.png")
    generate_figure(arr, stats, reg_results, fig_path, residual_depth=args.residual_depth)
    print(f"  {fig_path}")

    # Summary
    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"Image:       {args.image}")
    print(f"Regions:     {len(stats)}")
    print(f"κ_W cv:      {metrics['kappa_W']['cv']:.4f}")
    print(f"α cv:        {metrics['alpha']['cv']:.4f}")
    print(f"||Ω|| cv:     {metrics['omega_norm']['cv']:.4f}")
    print(f"Cumulative R²: {reg_results['cumulative_R2']:.4f}")


if __name__ == "__main__":
    main()
