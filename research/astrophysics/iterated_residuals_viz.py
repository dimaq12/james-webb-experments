import numpy as np
from PIL import Image
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import torch, sft_torch as sft
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, os; os.chdir('/home/dima/FA/astrophysics')

n, M = 10, 5; N = n * n; region_size = 80; stride = 80

basis_tensors = []
for d in range(M):
    B = np.zeros((N, N))
    for i in range(N):
        xi, yi = i % n, i // n
        j = yi * n + ((xi + d) % n)
        B[i, j] = 1.0; B[j, i] = 1.0
    basis_tensors.append(torch.from_numpy(B))
Bs = torch.stack(basis_tensors).to(torch.float64)

def compute_full_block(block):
    px = block[::block.shape[0]//n, ::block.shape[1]//n][:n, :n].ravel()
    px = px - np.min(px) + 0.1
    L = np.zeros((N, N))
    for i in range(N):
        xi, yi = i % n, i // n
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = xi+dx, yi+dy
            if 0 <= nx < n and 0 <= ny < n:
                j = ny*n + nx; L[i,j] -= 1; L[i,i] += 1
    A0 = torch.diag(torch.from_numpy(px)) + torch.from_numpy(L)*0.3
    prog = sft.operator(A0.to(torch.float64), Bs).compile("cpu")
    fam = prog.family
    tail = prog.tail(strict=False)
    kap = prog.kappa(method="perturbation", n_pert=3, eps=0.1)
    kw = kap.summary()['value']
    lam = fam.lam0; gaps = torch.diff(lam)
    W = fam.W; dec = prog.decompose(); sv = dec.singular
    return {
        'kw': float(kw),
        'alpha': float(tail.alpha if tail.alpha else 0),
        'class': tail.growth_class,
        'lam0_min': float(lam[0]), 'lam0_max': float(lam[-1]),
        'lam_std': float(lam.std()), 'lam_range': float(lam[-1] - lam[0]),
        'gap_median': float(gaps.median()), 'gap_min': float(gaps.min()),
        'gap_max': float(gaps.max()),
        'rank': fam.W_rank, 'cond': float(fam.condition_number()),
        'sv_max': float(sv[0]), 'sv_min': float(sv[-1]) if len(sv) > 0 else 0,
        'sv_std': float(sv.std()) if len(sv) > 0 else 0,
        'complexity': float(fam.complexity),
        'pixel_mean': float(block.mean()), 'pixel_std': float(block.std()),
    }

# ═══ SCAN DSCF1809 ═══
print("Loading DSCF1809.JPG...")
img = Image.open('imgs/DSCF1809.JPG').convert('L')
img = img.resize((img.width//4, img.height//4), Image.LANCZOS)
arr = np.array(img, dtype=np.float64) / 255.0
H, W = arr.shape
print(f'  {W}x{H}')

print('Scanning DSCF1809...')
t0 = time.perf_counter()
stats = []
for i in range(0, H - region_size, stride):
    if len(stats) >= 250: break
    for j in range(0, W - region_size, stride):
        if len(stats) >= 250: break
        block = arr[i:i+region_size, j:j+region_size]
        if block.max() - block.min() < 0.001: continue
        s = compute_full_block(block)
        s['i'] = i; s['j'] = j; s['ci'] = i + region_size//2; s['cj'] = j + region_size//2
        stats.append(s)
        if len(stats) % 50 == 0: print(f'  [{len(stats)}]')
print(f'  {len(stats)} regions in {(time.perf_counter()-t0):.0f}s')

# ═══ REGRESSION ═══
kw = np.array([s['kw'] for s in stats])
alpha = np.array([s['alpha'] for s in stats])
lam_std = np.array([s['lam_std'] for s in stats])
lam_range = np.array([s['lam_range'] for s in stats])
gap_median = np.array([s['gap_median'] for s in stats])
gap_min = np.array([s['gap_min'] for s in stats])
sv_std = np.array([s['sv_std'] for s in stats])
cond = np.array([s['cond'] for s in stats])
pm = np.array([s['pixel_mean'] for s in stats])
ps = np.array([s['pixel_std'] for s in stats])
gi = np.array([s['ci'] for s in stats]); gj = np.array([s['cj'] for s in stats])
gi_n = (gi - gi.mean())/gi.std(); gj_n = (gj - gj.mean())/gj.std()

lam0_min = np.array([s['lam0_min'] for s in stats]); lam0_max = np.array([s['lam0_max'] for s in stats])
gap_max = np.array([s['gap_max'] for s in stats])
sv_max = np.array([s['sv_max'] for s in stats]); sv_min = np.array([s['sv_min'] for s in stats])
complexity = np.array([s['complexity'] for s in stats])

def regress(y, X):
    c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ c
    resid = y - pred
    R2 = 1 - np.var(resid)/np.var(y) if np.var(y) > 0 else 0
    return pred, resid, R2

X0 = np.column_stack([pm, ps, np.ones_like(pm)])
pred0, resid0, R2_0 = regress(kw, X0)

X1 = np.column_stack([alpha, lam_std, lam_range, gap_median, gap_min, sv_std, cond,
                      gi_n, gj_n, gi_n**2, gj_n**2, gi_n*gj_n, np.ones_like(alpha)])
pred1, resid1, R2_1 = regress(resid0, X1)
cum1 = R2_0 + (1-R2_0)*R2_1

X2 = np.column_stack([lam0_min, lam0_max, gap_max, sv_max, sv_min, complexity, np.ones_like(lam0_min)])
pred2, resid2, R2_2 = regress(resid1, X2)
cum2 = R2_0 + (1-R2_0)*R2_1 + (1-R2_0)*(1-R2_1)*R2_2

# ═══ BUILD DENSE RESIDUAL MAPS ═══
def build_dense(vals, H, W):
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    dense = griddata((gi, gj), vals, (grid_y, grid_x), method='linear', fill_value=np.nan)
    mask = ~np.isnan(dense)
    dense[~mask] = 0
    dense = gaussian_filter(dense, sigma=2.5)
    dense[~mask] = np.nan
    return dense

kw_map = build_dense(kw, H, W)
res0_map = build_dense(resid0, H, W)
res1_map = build_dense(resid1, H, W)
res2_map = build_dense(resid2, H, W)

# ═══ JWST DATA ═══
jwst_res = np.load('spectral_scan_torch_results.npy', allow_pickle=True).item()
jk = jwst_res['stats']
jk_kw = np.array([s['kappa_W'] for s in jk])
jk_ps = np.array([s['pixel_std'] for s in jk]); jk_pm = np.array([s['pixel_mean'] for s in jk])
jk_alpha = np.array([s['tail_alpha'] for s in jk])
jk_gi = np.array([s['i'] for s in jk])+40; jk_gj = np.array([s['j'] for s in jk])+40
jk_gin = (jk_gi-jk_gi.mean())/jk_gi.std(); jk_gjn = (jk_gj-jk_gj.mean())/jk_gj.std()

X0j = np.column_stack([jk_pm, jk_ps, np.ones_like(jk_pm)])
pred0j, resid0j, R2_0j = regress(jk_kw, X0j)

X1j = np.column_stack([jk_alpha, jk_gin, jk_gjn, jk_gin**2, jk_gjn**2,
                       jk_gin*jk_gjn, np.ones_like(jk_alpha)])
pred1j, resid1j, R2_1j = regress(resid0j, X1j)
cum_jwst = R2_0j + (1-R2_0j)*R2_1j

# JWST image
hdul = fits.open('jw02736-o001_t001_nircam_clear-f090w_i2d.fits')
jwst_data = hdul['SCI'].data.astype(np.float64)
jwst_data[np.isnan(jwst_data)] = 0.0
Hj, Wj = jwst_data.shape
jwst_crop = jwst_data[:1100, :1100]
log_jwst = np.log10(jwst_crop - jwst_crop.min() + 0.01)

# Build JWST residual maps on full array then crop
def build_jwst(vals):
    grid_y, grid_x = np.mgrid[0:Hj, 0:Wj]
    dense = griddata((jk_gi, jk_gj), vals, (grid_y, grid_x), method='linear', fill_value=np.nan)
    mask = ~np.isnan(dense)
    dense[~mask] = 0
    dense = gaussian_filter(dense, sigma=2.5)
    dense[~mask] = np.nan
    return dense[:1100, :1100]

jwst_kw_map = build_jwst(jk_kw)
jwst_r0_map = build_jwst(resid0j)
jwst_r1_map = build_jwst(resid1j)

# ═══ FIGURE: 2 строки × 4 колонки ═══
# Row 1: DSCF1809 — original, κ_W, resid0, resid1+2
# Row 2: JWST — original, κ_W, resid0, resid1

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.22)

row_labels = ['DSCF1809 (фото)', 'JWST NIRCam F090W (телескоп)']
arrmaps = [arr, jwst_crop]
logmaps = [arr, log_jwst]
cmaps_img = ['gray', 'inferno']

for row_idx in range(2):
    # COL 0: Original
    ax = axes[row_idx, 0]
    if row_idx == 0:
        ax.imshow(arr, cmap='gray', origin='lower')
    else:
        valid = jwst_crop[jwst_crop > 0]
        vmin, vmax = np.percentile(valid, 1), np.percentile(valid, 99.5)
        ax.imshow(jwst_crop, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(f'{row_labels[row_idx]}\nИсходное изображение', fontsize=11, fontweight='bold')
    ax.set_xlabel('pixels'); ax.set_ylabel('pixels')

    # COL 1: κ_W map
    ax = axes[row_idx, 1]
    bg = arr if row_idx == 0 else log_jwst
    kwm = kw_map if row_idx == 0 else jwst_kw_map
    kw_arr = kw if row_idx == 0 else jk_kw
    ax.imshow(bg, cmap='gray', origin='lower', alpha=0.3,
              vmin=np.percentile(bg[bg>0] if row_idx==1 else bg, 20),
              vmax=np.percentile(bg[bg>0] if row_idx==1 else bg, 95))
    im = ax.imshow(kwm, cmap='plasma', origin='lower', alpha=0.7, interpolation='bilinear')
    ax.set_title(f'κ_W (спектральная кривизна)\n[{kw_arr.min():.0f}..{kw_arr.max():.0f}]',
                 fontsize=11, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.75); cbar.set_label('κ_W')

    # COL 2: Residual 0 (after brightness)
    ax = axes[row_idx, 2]
    r0m = res0_map if row_idx == 0 else jwst_r0_map
    r0_vals = resid0 if row_idx == 0 else resid0j
    R0_val = R2_0 if row_idx == 0 else R2_0j
    ax.imshow(bg, cmap='gray', origin='lower', alpha=0.5,
              vmin=np.percentile(bg[bg>0] if row_idx==1 else bg, 20),
              vmax=np.percentile(bg[bg>0] if row_idx==1 else bg, 95))
    vlim0 = max(abs(np.nanmin(r0m)), abs(np.nanmax(r0m))) if np.any(~np.isnan(r0m)) else 30
    im = ax.imshow(r0m, cmap='RdBu_r', origin='lower', alpha=0.65,
                   interpolation='bilinear', vmin=-vlim0, vmax=vlim0)
    ax.set_title(f'Остаток 0: после яркости\nR²={R0_val:.3f}  σ={r0_vals.std():.1f}',
                 fontsize=11, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.75); cbar.set_label('Residual')

    # COL 3: Residual 1 (or final)
    ax = axes[row_idx, 3]
    if row_idx == 0:
        # DSCF1809: show residual 1 AND residual 2 info
        r1m = res1_map
        r1_vals = resid1
        ax.imshow(arr, cmap='gray', origin='lower', alpha=0.5,
                  vmin=np.percentile(arr, 5), vmax=np.percentile(arr, 95))
        vlim1 = max(abs(np.nanmin(r1m)), abs(np.nanmax(r1m))) if np.any(~np.isnan(r1m)) else 30
        im = ax.imshow(r1m, cmap='RdBu_r', origin='lower', alpha=0.65,
                       interpolation='bilinear', vmin=-vlim1, vmax=vlim1)
        ax.set_title(f'Остаток 1: +α+спектр+поз.\n'
                     f'L1 R²={R2_1:.3f}  σ={r1_vals.std():.1f}\n'
                     f'Кум. R²={cum1:.3f}  (→L2: {cum2:.3f})',
                     fontsize=11, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, shrink=0.75); cbar.set_label('Residual')

        # DSCF1809 stats box
        dsc_info = (
            f"DSCF1809: {len(stats)} регионов\n"
            f"{'─'*22}\n"
            f"L0 яркость:      R²={R2_0:.3f} σ→{resid0.std():.0f}\n"
            f"L1 +α+спектр:    R²={R2_1:.3f} σ→{resid1.std():.0f}\n"
            f"L2 +gap+форма:   R²={R2_2:.3f} σ→{resid2.std():.0f}\n"
            f"{'─'*22}\n"
            f"ВСЕГО:           R²={cum2:.3f}\n"
            f"Cжатие σ:        {kw.std():.0f}→{resid2.std():.0f}\n"
            f"                   ({resid2.std()/kw.std()*100:.0f}% осталось)"
        )
        axes[0, 3].text(0.02, 0.98, dsc_info, transform=axes[0, 3].transAxes,
                        fontsize=7.5, verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='#fffde7', alpha=0.92,
                                  edgecolor='gray', linewidth=0.5))
    else:
        # JWST: residual 1
        r1m = jwst_r1_map
        r1_vals = resid1j
        ax.imshow(log_jwst, cmap='gray', origin='lower', alpha=0.5,
                  vmin=np.percentile(log_jwst, 20), vmax=np.percentile(log_jwst, 95))
        vlim1 = max(abs(np.nanmin(r1m)), abs(np.nanmax(r1m))) if np.any(~np.isnan(r1m)) else 30
        im = ax.imshow(r1m, cmap='RdBu_r', origin='lower', alpha=0.65,
                       interpolation='bilinear', vmin=-vlim1, vmax=vlim1)
        ax.set_title(f'Остаток 1: +α+позиция\n'
                     f'L1 R²={R2_1j:.3f}  σ={r1_vals.std():.1f}\n'
                     f'Кум. R²={cum_jwst:.3f}',
                     fontsize=11, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, shrink=0.75); cbar.set_label('Residual')

        jwst_info = (
            f"JWST: {len(jk)} регионов\n"
            f"{'─'*22}\n"
            f"L0 яркость:      R²={R2_0j:.3f} σ→{resid0j.std():.0f}\n"
            f"L1 +α+позиция:   R²={R2_1j:.3f} σ→{resid1j.std():.0f}\n"
            f"{'─'*22}\n"
            f"ВСЕГО:           R²={cum_jwst:.3f}\n"
            f"Сжатие σ:        {jk_kw.std():.0f}→{resid1j.std():.0f}\n"
            f"                   ({resid1j.std()/jk_kw.std()*100:.0f}% осталось)\n"
            f"\n⚠️ Только 50% объяснено!\n"
            f"   vs 93% для фото"
        )
        axes[1, 3].text(0.02, 0.98, jwst_info, transform=axes[1, 3].transAxes,
                        fontsize=7.5, verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='#ffe8e8', alpha=0.92,
                                  edgecolor='red', linewidth=0.8))

# ── ANNOTATIONS ON RESIDUAL MAPS ──
# DSCF1809 resid1: mark extremes
r1_clean = res1_map.copy(); r1_clean[np.isnan(r1_clean)] = 0
flat_idx = np.argsort(-r1_clean.ravel())
for rank, (label, color, x_off, y_off) in enumerate([
    ('+residual\n(сложнее\nожидаемого)', 'darkred', 300, 100),
    ('−residual\n(проще\nожидаемого)', 'darkblue', 250, -120),
]):
    for off in range(rank*300, (rank+1)*300):
        idx = flat_idx[off]
        cy, cx = idx // W, idx % W
        if not np.isnan(res1_map[cy, cx]) and 30<cy<H-30 and 30<cx<W-30:
            break
    axes[0, 3].annotate(label, xy=(cx, cy), fontsize=9, fontweight='bold', color=color,
                        xytext=(cx + x_off, cy + y_off),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2),
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), ha='center')

# JWST resid0: mark extreme unexplained
r0j_clean = jwst_r0_map.copy(); r0j_clean[np.isnan(r0j_clean)] = 0
flat_j = np.argsort(-r0j_clean.ravel())
for rank, (label, color, x_off, y_off) in enumerate([
    ('+', 'darkred', 200, 80), ('−', 'darkblue', 200, -80),
]):
    for off in range(rank*150, (rank+1)*150):
        idx = flat_j[off]
        cy, cx = idx // 1100, idx % 1100
        if not np.isnan(jwst_r0_map[cy, cx]) and 20<cy<1080 and 20<cx<1080:
            break
    axes[1, 2].annotate(label, xy=(cx, cy), fontsize=14, fontweight='bold', color=color,
                        xytext=(cx + x_off, cy + y_off),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.5),
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), ha='center')

# ── BOTTOM BAR ──
fig.text(0.5, 0.004,
         'Итерированная регрессия κ_W.  Колонки: исходное изображение → κ_W → остаток после яркости → остаток после спектра+позиции.  '
         'Фото сжимается до белого шума (93% объяснено), JWST сопротивляется (50% объяснено).  sft_torch.',
         ha='center', fontsize=12, style='italic',
         bbox=dict(boxstyle='round', facecolor='#eeeeee', alpha=0.9))

fig.suptitle('Итерированные остатки: визуализация скрытого спектрального поля\n'
             'DSCF1809 (фото, верх) vs JWST (телескоп, низ) — sft_torch',
             fontsize=15, fontweight='bold', y=0.99)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('iterated_residuals_viz.pdf', dpi=150, bbox_inches='tight')
fig.savefig('iterated_residuals_viz.png', dpi=150, bbox_inches='tight')
print('\nSaved iterated_residuals_viz.pdf + iterated_residuals_viz.png')
print(f'\nFinal stats:')
print(f'  DSCF1809: cumulative R² = {cum2:.4f}  (93% explained)')
print(f'  JWST:     cumulative R² = {cum_jwst:.4f}  (51% explained)')
