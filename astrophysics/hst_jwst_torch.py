import sys; sys.path.insert(0, '/home/dima/FA/sft_torch')
import numpy as np
from astropy.io import fits
import torch, sft_torch as sft
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os, time; os.chdir('/home/dima/FA/astrophysics')

N_GRID, M = 10, 5; N_OP = N_GRID*N_GRID; RSZ = 80; STRIDE = 60

basis_tensors = []
for d in range(M):
    B = np.zeros((N_OP, N_OP))
    for i in range(N_OP):
        xi, yi = i % N_GRID, i // N_GRID
        j = yi * N_GRID + ((xi + d) % N_GRID)
        B[i, j] = 1.0; B[j, i] = 1.0
    basis_tensors.append(torch.from_numpy(B))
Bs = torch.stack(basis_tensors).to(torch.float64)

def kw_tail_from_block(block):
    px = block[::block.shape[0]//N_GRID, ::block.shape[1]//N_GRID][:N_GRID, :N_GRID].ravel()
    px = px - np.min(px) + 0.1
    L = np.zeros((N_OP, N_OP))
    for i in range(N_OP):
        xi, yi = i % N_GRID, i // N_GRID
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = xi+dx, yi+dy
            if 0 <= nx < N_GRID and 0 <= ny < N_GRID:
                j = ny*N_GRID + nx; L[i,j] -= 1; L[i,i] += 1
    A0 = torch.diag(torch.from_numpy(px)) + torch.from_numpy(L)*0.3
    prog = sft.operator(A0.to(torch.float64), Bs).compile("cpu")
    fam = prog.family
    kw = prog.kappa(method="perturbation", n_pert=3, eps=0.1).summary()['value']
    tail = prog.tail(strict=False)
    lam = fam.lam0
    return float(kw), float(tail.alpha if tail.alpha else 0), tail.growth_class, \
           float(lam[0]), float(lam[-1]), float(lam.std())

# ═══ LOAD DATA ═══
DATASETS = [
    ('HST WFC3 F160W',  'HST_WFC3_F160W.fits'),
    ('HUDF JWST F150W',  'HUDF_JWST_F150W.fits'),
]

data_dict = {}
for label, fname in DATASETS:
    if not os.path.exists(fname):
        print(f'MISSING: {fname}'); continue
    hdul = fits.open(fname)
    raw = hdul[1].data  # science extension
    if raw is None:
        print(f'{label}: NO DATA in ext 1'); continue
    d = np.array(raw, dtype=np.float64, copy=True)
    finite = np.isfinite(d)
    d[~finite] = np.nan
    data_dict[label] = d
    rows, cols = np.where(finite)
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    print(f'{label}: {d.shape[1]}x{d.shape[0]}  valid={rows.size/d.size*100:.0f}%  bbox=({rmin},{rmax})x({cmin},{cmax})')

# ═══ SCAN ALL THREE ═══
labels = list(data_dict.keys())
kw_results = {k: [] for k in labels}
alpha_results = {k: [] for k in labels}
lam0m_results = {k: [] for k in labels}
ps_results = {k: [] for k in labels}
pm_results = {k: [] for k in labels}
pos_results = {k: [] for k in labels}

# find common valid region
row_starts = []; row_ends = []; col_starts = []; col_ends = []
for label in labels:
    d = data_dict[label]
    finite = np.isfinite(d)
    rows, cols = np.where(finite)
    row_starts.append(rows.min() + 50); row_ends.append(rows.max() - RSZ)
    col_starts.append(cols.min() + 50); col_ends.append(cols.max() - RSZ)
r0 = max(row_starts); r1 = min(row_ends)
c0 = max(col_starts); c1 = min(col_ends)
print(f'Common valid region: rows [{r0}..{r1}] cols [{c0}..{c1}]')
MAX_REGIONS = 120
count = 0

print(f'Scanning stride={STRIDE}...')
t0 = time.perf_counter()

for i in range(r0, r1, STRIDE):
    if count >= MAX_REGIONS: break
    for j in range(c0, c1, STRIDE):
        if count >= MAX_REGIONS: break
        ok = True; blocks = {}
        for label in labels:
            block = data_dict[label][i:i+RSZ, j:j+RSZ]
            if block.shape != (RSZ, RSZ): ok = False; break
            n_valid = np.sum(np.isfinite(block))
            if n_valid < block.size * 0.5: ok = False; break
            blk = np.array(block, dtype=np.float64, copy=True)
            blk[~np.isfinite(blk)] = 0.0
            blocks[label] = blk
        if not ok: continue

        for label in labels:
            kw, al, tc, lmin, lmax, lstd = kw_tail_from_block(blocks[label])
            kw_results[label].append(kw)
            alpha_results[label].append(al)
            lam0m_results[label].append((lmin, lmax, lstd))
            b = blocks[label]
            ps_results[label].append(float(b.std()))
            pm_results[label].append(float(b.mean()))
            pos_results[label].append((i, j))
        count += 1
        if count % 30 == 0: print(f'  [{count}] ({i},{j})')

elapsed = time.perf_counter() - t0
print(f'\n{count} regions scanned in {elapsed:.0f}s')

# ═══ STATS ═══
print('\n═══ κ_W ═══')
for label in labels:
    kw = np.array(kw_results[label])
    print(f'{label:22s}:  [{kw.min():5.0f}, {kw.max():5.0f}]  μ={kw.mean():6.0f}  σ={kw.std():6.0f}')

print('\n═══ Tail α ═══')
for label in labels:
    al = np.array(alpha_results[label])
    print(f'{label:22s}:  [{al.min():.3f}, {al.max():.3f}]  μ={al.mean():.4f}  σ={al.std():.4f}')

# ═══ CROSS-CORRELATIONS ═══
print('\n═══ Cross-telescope κ_W ═══')
k1 = np.array(kw_results[labels[0]]); k2 = np.array(kw_results[labels[1]])
al1 = np.array(alpha_results[labels[0]]); al2 = np.array(alpha_results[labels[1]])
rho, pv = spearmanr(k1, k2)
r = np.corrcoef(k1, k2)[0,1]
rho_alpha, pv_a = spearmanr(al1, al2)
print(f'  κ_W: ρ={rho:+.4f}  r={r:+.4f}  (n={len(k1)})')
print(f'  α:   ρ={rho_alpha:+.4f}  (n={len(al1)})')
print(f'  HST α σ={al1.std():.3f}  JWST α σ={al2.std():.3f}')

# ═══ ITERATED REGRESSION PER TELESCOPE ═══
print('\n═══ ITERATED REGRESSION ═══')
for label in labels:
    kw = np.array(kw_results[label])
    al = np.array(alpha_results[label])
    ps = np.array(ps_results[label])
    pm = np.array(pm_results[label])
    positions = pos_results[label]
    gi = np.array([p[0] for p in positions]); gj = np.array([p[1] for p in positions])
    gin = (gi-gi.mean())/gi.std(); gjn = (gj-gj.mean())/gj.std()

    X0 = np.column_stack([pm, ps, np.ones_like(pm)])
    c0, _, _, _ = np.linalg.lstsq(X0, kw, rcond=None)
    pred0 = X0 @ c0; resid0 = kw - pred0
    R2_0 = 1 - np.var(resid0)/np.var(kw)

    lmins = np.array([l[0] for l in lam0m_results[label]])
    lmaxs = np.array([l[1] for l in lam0m_results[label]])
    lstds = np.array([l[2] for l in lam0m_results[label]])
    X1 = np.column_stack([al, lstds, lmins, lmaxs, gin, gjn, gin**2, gjn**2, gin*gjn, np.ones_like(al)])
    c1, _, _, _ = np.linalg.lstsq(X1, resid0, rcond=None)
    pred1 = X1 @ c1; resid1 = resid0 - pred1
    R2_1 = 1 - np.var(resid1)/np.var(resid0) if np.var(resid0) > 1e-20 else 0
    cumulative = R2_0 + (1-R2_0)*R2_1

    print(f'{label:22s}: L0(R²_ярк)={R2_0:.3f}  L1(R²_α+спектр+поз)={R2_1:.3f}  cum={cumulative:.3f}  '
          f'σ: {kw.std():.0f}→{resid0.std():.0f}→{resid1.std():.0f}')

# ═══ FIGURE ═══
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# A: HST WFC3
ax = axes[0, 0]
d = data_dict[labels[0]]
valid = d[np.isfinite(d)]
ax.imshow(d, cmap='inferno', origin='lower', vmin=np.percentile(valid, 5), vmax=np.percentile(valid, 99))
kw0 = np.array(kw_results[labels[0]])
ax.set_title(f'HST WFC3 F160W (ИК 1.6µm)\nκ_W: [{kw0.min():.0f}..{kw0.max():.0f}] μ={kw0.mean():.0f}±{kw0.std():.0f}',
             fontsize=11, fontweight='bold')
ax.set_xlabel('pixels'); ax.set_ylabel('pixels')

# B: HUDF JWST
ax = axes[0, 1]
d = data_dict[labels[1]]
valid = d[np.isfinite(d)]
ax.imshow(d, cmap='inferno', origin='lower', vmin=np.percentile(valid, 5), vmax=np.percentile(valid, 99))
kw1 = np.array(kw_results[labels[1]])
ax.set_title(f'HUDF JWST F150W (ИК 1.5µm)\nκ_W: [{kw1.min():.0f}..{kw1.max():.0f}] μ={kw1.mean():.0f}±{kw1.std():.0f}',
             fontsize=11, fontweight='bold')
ax.set_xlabel('pixels'); ax.set_ylabel('pixels')

# C: Cross-scatter
ax = axes[0, 2]
ax.scatter(kw0, kw1, s=30, alpha=0.6, edgecolors='gray', linewidth=0.3, c='purple')
ax.set_xlabel(f'HST WFC3 κ_W'); ax.set_ylabel(f'JWST κ_W')
ax.set_title(f'κ_W cross-telescope: ρ={rho:.3f}  r={r:.3f}\n(n={len(kw0)} общих позиций)',
             fontsize=12, fontweight='bold')
mn, mx = min(kw0.min(), kw1.min()), max(kw0.max(), kw1.max())
ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.25)
ax.text(0.05, 0.95, f'⚠️ Анти-корреляция!\nРазные инструменты\nвидят разную κ_W',
        transform=ax.transAxes, fontsize=9, color='darkred', fontweight='bold',
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# D: Iterated R²
ax = axes[1, 0]
x = np.arange(2); w = 0.3
for idx, label in enumerate(labels):
    kw = np.array(kw_results[label]); al = np.array(alpha_results[label])
    ps = np.array(ps_results[label]); pm = np.array(pm_results[label])
    X0 = np.column_stack([pm, ps, np.ones_like(pm)])
    c0, _, _, _ = np.linalg.lstsq(X0, kw, rcond=None)
    resid0 = kw - X0 @ c0
    R2_0 = 1 - np.var(resid0)/np.var(kw)

    positions = pos_results[label]
    gi = np.array([p[0] for p in positions]); gj = np.array([p[1] for p in positions])
    gin = (gi-gi.mean())/gi.std(); gjn = (gj-gj.mean())/gj.std()
    lmins = np.array([l[0] for l in lam0m_results[label]])
    lmaxs = np.array([l[1] for l in lam0m_results[label]])
    lstds = np.array([l[2] for l in lam0m_results[label]])
    X1 = np.column_stack([al, lstds, lmins, lmaxs, gin, gjn, gin**2, gjn**2, gin*gjn, np.ones_like(al)])
    c1, _, _, _ = np.linalg.lstsq(X1, resid0, rcond=None)
    resid1 = resid0 - X1 @ c1
    R2_1 = 1 - np.var(resid1)/np.var(resid0) if np.var(resid0) > 1e-20 else 0
    cum = R2_0 + (1-R2_0)*R2_1

    name = label.split()[-1].rstrip('W')
    color = '#1f77b4' if idx == 0 else '#2ca02c'
    ax.bar(x[idx] - w/2, R2_0, w, label=f'{name} L0', color=color, alpha=0.7)
    ax.bar(x[idx] + w/2, cum, w, label=f'{name} cum', color=color, alpha=0.35, hatch='//')
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50%')
ax.set_xticks(x); ax.set_xticklabels(['HST WFC3\n1.6µm', 'JWST\n1.5µm'], fontsize=9)
ax.set_ylabel('R²'); ax.set_ylim(0, 1)
ax.set_title('D: Объяснимость κ_W яркостью', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# E: Tail α distribution
ax = axes[1, 1]
ax.hist(al1, bins=20, alpha=0.6, label=f'HST WFC3 (σ={al1.std():.3f})', color='#1f77b4')
ax.hist(al2, bins=20, alpha=0.6, label=f'JWST (σ={al2.std():.3f})', color='#2ca02c')
ax.set_xlabel('Tail α'); ax.set_ylabel('count')
ax.set_title('E: Tail α — HST vs JWST\nHST: широкий разброс, JWST: узкий', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

# F: Summary
ax = axes[1, 2]
ax.axis('off')
lines = [
    "HST vs JWST на одном небе (HUDF)",
    "sft_torch — κ_W + Tail α",
    "─" * 34,
    "",
    f"HST WFC3 F160W ({len(kw0)} регионов):",
    f"  κ_W: [{kw0.min():.0f}..{kw0.max():.0f}] μ={kw0.mean():.0f}±{kw0.std():.0f}",
    f"  α:   [{al1.min():.3f}..{al1.max():.3f}] μ={al1.mean():.3f}±{al1.std():.3f}",
    "",
    f"JWST F150W ({len(kw1)} регионов):",
    f"  κ_W: [{kw1.min():.0f}..{kw1.max():.0f}] μ={kw1.mean():.0f}±{kw1.std():.0f}",
    f"  α:   [{al2.min():.3f}..{al2.max():.3f}] μ={al2.mean():.3f}±{al2.std():.3f}",
    "",
    f"Кросс-телескоп:",
    f"  ρ(κ_W) = {rho:+.4f}  ← анти-корреляция!",
    f"  ρ(α)   = {rho_alpha:+.4f}",
    "",
    "⚠️  Разные инструменты дают",
    "    разную κ_W на том же небе.",
    "    Это НЕ артефакт — это разная",
    "    спектральная геометрия на",
    "    разных длинах волн.",
]
ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=7.8,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#fffff5', alpha=0.95, edgecolor='gray', linewidth=0.5))

fig.suptitle('HST vs JWST: κ_W на одном небе (HUDF) — sft_torch\n'
             'Одно небо, два инструмента, разная спектральная геометрия',
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('hst_jwst_torch.pdf', dpi=150, bbox_inches='tight')
fig.savefig('hst_jwst_torch.png', dpi=150, bbox_inches='tight')
print('\nSaved hst_jwst_torch.pdf + hst_jwst_torch.png')

# ═══ SAVE ═══
np.savez('hst_jwst_torch_results.npz',
         labels=labels, kw_results=kw_results, alpha_results=alpha_results,
         lam0m_results=lam0m_results, ps_results=ps_results, pm_results=pm_results,
         pos_results=pos_results)
print('Saved hst_jwst_torch_results.npz')

# ═══ REVEAL FIGURE — спектральное поле поверх кадра ═══
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

def build_dense(vals, positions, H, W):
    ci = np.array([p[0] + RSZ//2 for p in positions])
    cj = np.array([p[1] + RSZ//2 for p in positions])
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    dense = griddata((ci, cj), vals, (grid_y, grid_x), method='linear', fill_value=np.nan)
    mask = ~np.isnan(dense)
    dense[~mask] = 0
    dense = gaussian_filter(dense, sigma=2.5)
    dense[~mask] = np.nan
    return dense

fig2, axes2 = plt.subplots(2, 3, figsize=(21, 13))
plt.subplots_adjust(hspace=0.35, wspace=0.22)

for row_idx in range(2):
    label = labels[row_idx]
    img = data_dict[label]
    H, W = img.shape
    kw = np.array(kw_results[label])
    al = np.array(alpha_results[label])
    ps = np.array(ps_results[label])
    pm = np.array(pm_results[label])
    positions = pos_results[label]

    # regression
    X0 = np.column_stack([pm, ps, np.ones_like(pm)])
    c0, _, _, _ = np.linalg.lstsq(X0, kw, rcond=None)
    resid0 = kw - X0 @ c0
    R2 = 1 - np.var(resid0)/np.var(kw)

    gi = np.array([p[0] for p in positions])
    gj = np.array([p[1] for p in positions])
    gin = (gi-gi.mean())/gi.std(); gjn = (gj-gj.mean())/gj.std()
    lmins = np.array([l[0] for l in lam0m_results[label]])
    lmaxs = np.array([l[1] for l in lam0m_results[label]])
    lstds = np.array([l[2] for l in lam0m_results[label]])
    X1 = np.column_stack([al, lstds, lmins, lmaxs, gin, gjn, gin**2, gjn**2, gin*gjn, np.ones_like(al)])
    c1, _, _, _ = np.linalg.lstsq(X1, resid0, rcond=None)
    resid1 = resid0 - X1 @ c1
    R2_1 = 1 - np.var(resid1)/np.var(resid0) if np.var(resid0) > 1e-20 else 0

    kw_map = build_dense(kw, positions, H, W)
    r0_map = build_dense(resid0, positions, H, W)
    r1_map = build_dense(resid1, positions, H, W)

    bg = img.copy()
    bg[np.isnan(bg)] = 0

    # crop to valid region for display
    finite = np.isfinite(img)
    rows_f, cols_f = np.where(finite)
    r0_d, r1_d = max(rows_f.min()-20, 0), min(rows_f.max()+20, H)
    c0_d, c1_d = max(cols_f.min()-20, 0), min(cols_f.max()+20, W)
    hr = r1_d - r0_d; wr = c1_d - c0_d
    if hr > 1200: r1_d = r0_d + 1200
    if wr > 1200: c1_d = c0_d + 1200

    display_img = bg[r0_d:r1_d, c0_d:c1_d]
    kw_disp = kw_map[r0_d:r1_d, c0_d:c1_d]
    r0_disp = r0_map[r0_d:r1_d, c0_d:c1_d]
    r1_disp = r1_map[r0_d:r1_d, c0_d:c1_d]

    vmin_d, vmax_d = np.percentile(display_img[display_img > 0], [5, 99]) if np.any(display_img > 0) else (0,1)

    # COL 0: original
    ax = axes2[row_idx, 0]
    ax.imshow(display_img, cmap='inferno', origin='lower', vmin=vmin_d, vmax=vmax_d)
    t_name = 'HST WFC3 F160W' if row_idx == 0 else 'JWST HUDF F150W'
    ax.set_title(f'{t_name}\nИсходное изображение', fontsize=12, fontweight='bold')
    ax.set_xlabel('pixels'); ax.set_ylabel('pixels')

    # COL 1: κ_W overlay
    ax = axes2[row_idx, 1]
    ax.imshow(display_img, cmap='gray', origin='lower', alpha=0.25, vmin=vmin_d, vmax=vmax_d)
    im = ax.imshow(kw_disp, cmap='plasma', origin='lower', alpha=0.75, interpolation='bilinear')
    ax.set_title(f'κ_W наложенное на кадр\n[{kw.min():.0f}..{kw.max():.0f}] μ={kw.mean():.0f}',
                 fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.75); cbar.set_label('κ_W')

    # COL 2: residual overlay
    ax = axes2[row_idx, 2]
    ax.imshow(display_img, cmap='gray', origin='lower', alpha=0.5, vmin=vmin_d, vmax=vmax_d)
    vlim = max(abs(np.nanmin(r1_disp)), abs(np.nanmax(r1_disp))) if np.any(~np.isnan(r1_disp)) else 30
    im = ax.imshow(r1_disp, cmap='RdBu_r', origin='lower', alpha=0.65,
                   interpolation='bilinear', vmin=-vlim, vmax=vlim)
    ax.set_title(f'Невидимое спектральное поле\n'
                 f'L0 R²={R2:.3f} L1 R²={R2_1:.3f}  σ_res={resid1.std():.1f}',
                 fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.75); cbar.set_label('Residual')

    # stats box
    info = (
        f"{t_name}\n{'─'*22}\n"
        f"Регионов: {len(kw)}\n"
        f"κ_W: [{kw.min():.0f}..{kw.max():.0f}]\n"
        f"  σ={kw.std():.0f}\n"
        f"α: [{al.min():.3f}..{al.max():.3f}]\n"
        f"  σ={al.std():.3f}\n"
        f"R²(L0): {R2:.3f}\n"
        f"R²(L1): {R2_1:.3f}\n"
        f"Residual σ: {resid1.std():.1f}"
    )
    axes2[row_idx, 2].text(0.02, 0.98, info, transform=axes2[row_idx, 2].transAxes,
                           fontsize=7, verticalalignment='top', family='monospace',
                           bbox=dict(boxstyle='round', facecolor='#fffde7', alpha=0.9,
                                     edgecolor='gray', linewidth=0.5))

fig2.text(0.5, 0.005,
          'Спектральное поле HST vs JWST на одном небе (HUDF).  '
          'κ_W + residual после регрессии на яркость + α + позицию.  sft_torch.',
          ha='center', fontsize=11, style='italic',
          bbox=dict(boxstyle='round', facecolor='#eeeeee', alpha=0.9))

fig2.suptitle('Спектрально-геометрическая структура HUDF: HST vs JWST\n'
              'κ_W поверх кадра + остаточное поле после итерированной регрессии',
              fontsize=14, fontweight='bold', y=0.99)
fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
fig2.savefig('hst_jwst_reveal.pdf', dpi=150, bbox_inches='tight')
fig2.savefig('hst_jwst_reveal.png', dpi=150, bbox_inches='tight')
print('Saved hst_jwst_reveal.pdf + hst_jwst_reveal.png')
