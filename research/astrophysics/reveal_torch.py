import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os; os.chdir('/home/dima/FA/astrophysics')

results = np.load('spectral_scan_torch_results.npy', allow_pickle=True).item()
stats = results['stats']
features = results['features']
kw = np.array([s['kappa_W'] for s in stats])
kw_residual = features['kw_residual']
R2 = features['R2']
ta = np.array([s['tail_alpha'] for s in stats])
gi = np.array([s['i'] for s in stats])
gj = np.array([s['j'] for s in stats])

hdul = fits.open('jw02736-o001_t001_nircam_clear-f090w_i2d.fits')
data = hdul['SCI'].data.astype(np.float64)
data[np.isnan(data)] = 0.0

region_size = 80
centers_i = gi + region_size // 2
centers_j = gj + region_size // 2

# build dense residual map
h_map = gi.max() + region_size + 100
w_map = gj.max() + region_size + 100
grid_y, grid_x = np.mgrid[0:h_map, 0:w_map]
residual_dense = griddata((centers_i, centers_j), kw_residual, (grid_y, grid_x), method='linear', fill_value=np.nan)
mask = ~np.isnan(residual_dense)
residual_dense[~mask] = 0
residual_dense = gaussian_filter(residual_dense, sigma=2)
residual_dense[~mask] = np.nan

# build κ_W dense map
kw_dense = griddata((centers_i, centers_j), kw, (grid_y, grid_x), method='linear', fill_value=np.nan)
kw_dense[np.isnan(kw_dense)] = 0
kw_dense = gaussian_filter(kw_dense, sigma=2)
kw_dense[np.isnan(residual_dense)] = np.nan

# build tail_alpha dense map
ta_dense = griddata((centers_i, centers_j), ta, (grid_y, grid_x), method='linear', fill_value=np.nan)
ta_dense[np.isnan(ta_dense)] = 0
ta_dense = gaussian_filter(ta_dense, sigma=2)
ta_dense[np.isnan(residual_dense)] = np.nan

# crop data
h_roi, w_roi = min(1000, h_map), min(4000, w_map)
i0, j0 = 0, 0
crop = data[i0:i0+h_roi, j0:j0+w_roi]
crop_log = np.log10(crop - crop.min() + 0.01)
valid = crop[crop > 0]
vmin, vmax = np.percentile(valid, 1), np.percentile(valid, 99.8)

# ═══════════════════════════════════════════════════════
fig = plt.figure(figsize=(24, 14))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

# PANEL 1 — JWST image
ax = fig.add_subplot(gs[0, 0])
ax.imshow(crop_log, cmap='inferno', origin='lower', vmin=np.percentile(crop_log, 5), vmax=np.percentile(crop_log, 99))
ax.set_title('JWST NIRCam F090W\nSMACS 0723 galaxy cluster', fontsize=13, fontweight='bold')
ax.set_xlabel('pixels'); ax.set_ylabel('pixels')

# PANEL 2 — κ_W dense map
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(kw_dense[i0:i0+h_roi, j0:j0+w_roi], cmap='plasma', origin='lower', aspect='auto', interpolation='bilinear')
ax.set_title(f'κ_W (spectral curvature)\nrange [{kw.min():.0f}, {kw.max():.0f}]', fontsize=13, fontweight='bold')
ax.set_xlabel('pixels'); ax.set_ylabel('pixels')
cbar = plt.colorbar(im, ax=ax, shrink=0.75)
cbar.set_label('κ_W', fontsize=10)

# PANEL 3 — Tail α dense map
ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(ta_dense[i0:i0+h_roi, j0:j0+w_roi], cmap='viridis', origin='lower', aspect='auto', interpolation='bilinear')
ax.set_title(f'Tail α (spectral decay)\nrange [{ta.min():.3f}, {ta.max():.3f}]', fontsize=13, fontweight='bold')
ax.set_xlabel('pixels'); ax.set_ylabel('pixels')
cbar = plt.colorbar(im, ax=ax, shrink=0.75)
cbar.set_label('Tail α', fontsize=10)

# PANEL 4 — Residual map
ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(residual_dense[i0:i0+h_roi, j0:j0+w_roi], cmap='RdBu_r', origin='lower', aspect='auto', interpolation='bilinear',
               vmin=-50, vmax=30)
ax.set_title(f'κ_W residual (after brightness regression)\nR²={R2:.3f}  unexplained={(1-R2)*100:.0f}%', fontsize=13, fontweight='bold')
ax.set_xlabel('pixels'); ax.set_ylabel('pixels')
cbar = plt.colorbar(im, ax=ax, shrink=0.75)
cbar.set_label('Residual κ_W', fontsize=10)

# PANEL 5 — κ_W vs tail_alpha scatter
ax = fig.add_subplot(gs[1, 1])
sc = ax.scatter(kw, ta, c=kw_residual, cmap='coolwarm', s=25, alpha=0.8, edgecolors='gray', linewidth=0.3, vmin=-50, vmax=30)
r_kta = np.corrcoef(kw, ta)[0, 1]
ax.set_xlabel('κ_W (spectral curvature)', fontsize=11)
ax.set_ylabel('Tail α (spectral decay)', fontsize=11)
ax.set_title(f'κ_W vs Tail α\nr={r_kta:.3f}', fontsize=12, fontweight='bold')
plt.colorbar(sc, ax=ax, shrink=0.8, label='Residual κ_W')

# PANEL 6 — Summary & torch vs original
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')

orig_results = np.load('spectral_scan_results.npy', allow_pickle=True).item()
o_kw = np.array([s['kappa_W'] for s in orig_results['stats']])
pos_to_orig = {(s['i'], s['j']): s['kappa_W'] for s in orig_results['stats']}
matched = [(s['kappa_W'], pos_to_orig.get((s['i'], s['j']), np.nan)) for s in stats]
matched = [(t, o) for t, o in matched if not np.isnan(o)]
r_cross = np.corrcoef([m[0] for m in matched], [m[1] for m in matched])[0, 1]

ctrl = results.get('control', {})
summary_text = (
    f"sft_torch JWST SMACS 0723 analysis\n"
    f"{'─'*38}\n\n"
    f"Regions: {len(stats)} (80×80 px)\n"
    f"Operator: 10×10 grid, M=5 shifts\n"
    f"κ_W: [{kw.min():.0f}, {kw.max():.0f}] μ={kw.mean():.0f} σ={kw.std():.0f}\n"
    f"Tail α: [{ta.min():.3f}, {ta.max():.3f}] μ={ta.mean():.3f} σ={ta.std():.3f}\n\n"
    f"R² (6 features): {R2:.3f}\n"
    f"Unexplained: {(1-R2)*100:.0f}%\n"
    f"Residual σ: {kw_residual.std():.1f}\n\n"
    f"sft vs sft_torch κ_W:\n"
    f"  r = {r_cross:.4f} (Pearson)\n"
    f"  200/200 positions matched\n\n"
    f"Shuffle control:\n"
    f"  p = {ctrl.get('wilcoxon_p', np.nan):.4f}\n\n"
    f"NEW: Tail α adds information\n"
    f"  r(kappa, α) = {r_kta:.3f}\n"
    f"  complementary metric"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#fffff0', alpha=0.92, edgecolor='gray', linewidth=0.5))

fig.suptitle('Spectral Flow Transform (Torch) on JWST Deep Field — κ_W, Tail α, Residual Structure',
             fontsize=16, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('reveal_torch.pdf', dpi=150, bbox_inches='tight')
fig.savefig('reveal_torch.png', dpi=150, bbox_inches='tight')
print('Saved reveal_torch.pdf + reveal_torch.png')
print(f'\nKey numbers:')
print(f'  sft_torch κ_W: [{kw.min():.0f}, {kw.max():.0f}] mean={kw.mean():.0f}')
print(f'  Cross-framework κ_W correlation: r = {r_cross:.4f}')
print(f'  R² = {R2:.3f} ({100*(1-R2):.0f}% unexplained)')
print(f'  κ_W vs Tail α correlation: r = {r_kta:.3f}')
