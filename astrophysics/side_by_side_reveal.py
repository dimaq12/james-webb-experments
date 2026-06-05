import sys; sys.path.insert(0, '/home/dima/FA/sft_torch')
import numpy as np
from PIL import Image
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import torch, sft_torch as sft
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os; os.chdir('/home/dima/FA/astrophysics')

# ═══ PARAMETERS ═══
n, M = 10, 5; N = n * n; region_size = 80; stride = 100

basis_tensors = []
for d in range(M):
    B = np.zeros((N, N))
    for i in range(N):
        xi, yi = i % n, i // n
        j = yi * n + ((xi + d) % n)
        B[i, j] = 1.0; B[j, i] = 1.0
    basis_tensors.append(torch.from_numpy(B))
Bs = torch.stack(basis_tensors).to(torch.float64)

def compute_kw_block(block, rng_seed):
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
    tail = prog.tail(strict=False)
    result = prog.kappa(method="perturbation", n_pert=3, eps=0.1)
    kw = result.summary()['value']
    return float(kw), float(tail.alpha if tail.alpha else 0)

def scan_image_dense(arr, max_regions=300):
    H, W = arr.shape
    stats = []
    for i in range(0, H - region_size, stride):
        for j in range(0, W - region_size, stride):
            if len(stats) >= max_regions: break
            block = arr[i:i+region_size, j:j+region_size]
            if np.count_nonzero(block) < block.size * 0.5: continue
            kw, alpha = compute_kw_block(block, len(stats))
            stats.append({'i': i, 'j': j, 'kw': kw, 'alpha': alpha,
                          'center_i': i + region_size//2, 'center_j': j + region_size//2})
        if len(stats) >= max_regions: break
    return stats

# ═══ LOAD DATA ═══
print("Loading data...")

# Regular photo
img = Image.open('imgs/IMG_2193.JPG').convert('L')
img = img.resize((img.width//4, img.height//4), Image.LANCZOS)
photo_arr = np.array(img, dtype=np.float64) / 255.0
Hp, Wp = photo_arr.shape
print(f'Photo: {Wp}x{Hp}')

# JWST
hdul = fits.open('jw02736-o001_t001_nircam_clear-f090w_i2d.fits')
jwst_arr = hdul['SCI'].data.astype(np.float64)
jwst_arr[np.isnan(jwst_arr)] = 0.0
Hj, Wj = jwst_arr.shape

# ═══ SCAN PHOTO ═══
print("Scanning photo...")
photo_stats = scan_image_dense(photo_arr, max_regions=250)
n_photo = len(photo_stats)
print(f'  {n_photo} regions scanned')

# ═══ PREDICT κ_W FROM PIXEL FEATURES ═══
# regress κ_W on pixel_mean + pixel_std
photo_kw = np.array([s['kw'] for s in photo_stats])
photo_alpha = np.array([s['alpha'] for s in photo_stats])

pixel_means = np.array([photo_arr[s['i']:s['i']+region_size, s['j']:s['j']+region_size].mean() for s in photo_stats])
pixel_stds = np.array([photo_arr[s['i']:s['i']+region_size, s['j']:s['j']+region_size].std() for s in photo_stats])

X_photo = np.column_stack([pixel_means, pixel_stds, np.ones_like(pixel_means)])
coeffs_photo, _, _, _ = np.linalg.lstsq(X_photo, photo_kw, rcond=None)
kw_pred_photo = X_photo @ coeffs_photo
kw_resid_photo = photo_kw - kw_pred_photo
R2_photo = 1 - np.var(kw_resid_photo) / np.var(photo_kw)

# ═══ JWST (reuse saved) ═══
jwst_res = np.load('spectral_scan_torch_results.npy', allow_pickle=True).item()
jwst_stats = jwst_res['stats']
jwst_kw = np.array([s['kappa_W'] for s in jwst_stats])
jwst_kw_pred = jwst_res['features']['kw_pred']
jwst_kw_resid = jwst_res['features']['kw_residual']
jwst_alpha = np.array([s['tail_alpha'] for s in jwst_stats])
jwst_R2 = jwst_res['features']['R2']

# ═══ BUILD DENSE RESIDUAL MAPS ═══
def build_residual_map(stats_arr, resid_vals, H, W, region_size):
    centers_i = np.array([s['i'] + region_size//2 for s in stats_arr])
    centers_j = np.array([s['j'] + region_size//2 for s in stats_arr])
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    dense = griddata((centers_i, centers_j), resid_vals, (grid_y, grid_x),
                     method='linear', fill_value=np.nan)
    mask = ~np.isnan(dense)
    dense[~mask] = 0
    dense = gaussian_filter(dense, sigma=region_size//20)
    dense[~mask] = np.nan
    return dense

print("Building residual maps...")
photo_resid_map = build_residual_map(photo_stats, kw_resid_photo, Hp, Wp, region_size)
jwst_resid_map = build_residual_map(jwst_stats, jwst_kw_resid, Hj, Wj, region_size)

# Crop JWST to a reasonable region
jwst_crop = jwst_arr[:1200, :1200].copy()
jwst_resid_crop = jwst_resid_map[:1200, :1200]

# ═══ ALSO BUILD κ_W DENSE MAP ═══
photo_kw_map = build_residual_map(photo_stats, photo_kw, Hp, Wp, region_size)
jwst_kw_map = build_residual_map(jwst_stats, jwst_kw, Hj, Wj, region_size)
jwst_kw_crop = jwst_kw_map[:1200, :1200]

# ═══ FIGURE ═══
# 2 rows x 3 cols:
# Row 1: regular photo
# Row 2: JWST
# Col 1: original image, Col 2: κ_W overlay, Col 3: residual overlay

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# ── ROW 1: PHOTO ──
# A1: original photo
ax = axes[0, 0]
ax.imshow(photo_arr, cmap='gray', origin='lower')
ax.set_title('A: Обычное фото\n(то, что видит камера)', fontsize=13, fontweight='bold')
ax.set_xlabel('pixels'); ax.set_ylabel('pixels')

# B1: κ_W map
ax = axes[0, 1]
ax.imshow(photo_arr, cmap='gray', origin='lower', alpha=0.3)
im = ax.imshow(photo_kw_map, cmap='plasma', origin='lower', alpha=0.7, interpolation='bilinear')
ax.set_title(f'B: κ_W (спектральная кривизна)\n[{photo_kw.min():.0f}..{photo_kw.max():.0f}]  R²={R2_photo:.2f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('pixels')
cbar = plt.colorbar(im, ax=ax, shrink=0.75)
cbar.set_label('κ_W', fontsize=9)

# C1: residual overlay (hidden field)
ax = axes[0, 2]
ax.imshow(photo_arr, cmap='gray', origin='lower',
          vmin=np.percentile(photo_arr, 5), vmax=np.percentile(photo_arr, 95), alpha=0.5)
vlim = max(abs(photo_resid_map[~np.isnan(photo_resid_map)].min()),
            abs(photo_resid_map[~np.isnan(photo_resid_map)].max())) if np.any(~np.isnan(photo_resid_map)) else 30
im = ax.imshow(photo_resid_map, cmap='RdBu_r', origin='lower', alpha=0.6,
               interpolation='bilinear', vmin=-vlim, vmax=vlim)
ax.set_title(f'C: Невидимое спектральное поле\n(κ_W residual — что камера НЕ видит)\n'
             f'σ_resid={kw_resid_photo.std():.1f}  необъяснено={(1-R2_photo)*100:.0f}%',
             fontsize=13, fontweight='bold')
ax.set_xlabel('pixels')
cbar = plt.colorbar(im, ax=ax, shrink=0.75)
cbar.set_label('Residual κ_W', fontsize=9)

# ── ROW 2: JWST ──
# A2: JWST
ax = axes[1, 0]
valid = jwst_crop[jwst_crop > 0]
vmin_j, vmax_j = np.percentile(valid, 1), np.percentile(valid, 99.5)
ax.imshow(jwst_crop, cmap='inferno', origin='lower', vmin=vmin_j, vmax=vmax_j)
ax.set_title('D: JWST NIRCam F090W\n(то, что видит телескоп)', fontsize=13, fontweight='bold')
ax.set_xlabel('pixels'); ax.set_ylabel('pixels')

# B2: JWST κ_W map
ax = axes[1, 1]
bg = np.log10(jwst_crop - jwst_crop.min() + 0.01)
ax.imshow(bg, cmap='gray', origin='lower', alpha=0.3,
          vmin=np.percentile(bg, 20), vmax=np.percentile(bg, 95))
im = ax.imshow(jwst_kw_crop, cmap='plasma', origin='lower', alpha=0.7, interpolation='bilinear')
ax.set_title(f'E: JWST κ_W (спектральная кривизна)\n'
             f'[{jwst_kw.min():.0f}..{jwst_kw.max():.0f}]  R²={jwst_R2:.2f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('pixels')
cbar = plt.colorbar(im, ax=ax, shrink=0.75)
cbar.set_label('κ_W', fontsize=9)

# C2: JWST residual
ax = axes[1, 2]
ax.imshow(bg, cmap='gray', origin='lower',
          vmin=np.percentile(bg, 20), vmax=np.percentile(bg, 95), alpha=0.5)
vlim_j = 60
im = ax.imshow(jwst_resid_crop, cmap='RdBu_r', origin='lower', alpha=0.6,
               interpolation='bilinear', vmin=-vlim_j, vmax=vlim_j)
ax.set_title(f'F: Невидимое спектральное поле JWST\n(κ_W residual — структура за пределами яркости)\n'
             f'σ_resid={jwst_kw_resid.std():.1f}  необъяснено={(1-jwst_R2)*100:.0f}%',
             fontsize=13, fontweight='bold')
ax.set_xlabel('pixels')
cbar = plt.colorbar(im, ax=ax, shrink=0.75)
cbar.set_label('Residual κ_W', fontsize=9)

# ── ANNOTATIONS ──
# Photo annotations — find interesting regions
resid_photo_clean = photo_resid_map.copy()
resid_photo_clean[np.isnan(resid_photo_clean)] = 0
flat_idx = np.argsort(-resid_photo_clean.ravel())
for rank, (label, color) in enumerate([
    ('Высокая сложность', 'darkred'),
    ('Низкая сложность', 'darkblue'),
]):
    for offset in range(rank * 500, (rank+1)*500):
        idx = flat_idx[offset]
        cy, cx = idx // Wp, idx % Wp
        if 20 < cy < Hp-20 and 20 < cx < Wp-20:
            val = photo_resid_map[cy, cx]
            if not np.isnan(val):
                break
    else:
        continue
    axes[0, 2].annotate(f'{label}\nres={val:+.0f}', xy=(cx, cy),
                        fontsize=10, fontweight='bold', color=color,
                        xytext=(cx + 200, cy + 100 if rank == 0 else cy - 150),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2),
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
                        ha='center')

# JWST annotations
axes[1, 2].annotate(
    'Спектрально проще\nчем предсказано',
    xy=(80, 80), fontsize=9, fontweight='bold', color='darkblue',
    xytext=(250, 30),
    arrowprops=dict(arrowstyle='->', color='darkblue', lw=2),
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), ha='center')

axes[1, 2].annotate(
    'Спектрально сложнее\nчем предсказано',
    xy=(620, 580), fontsize=9, fontweight='bold', color='darkred',
    xytext=(850, 700),
    arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), ha='center')

# ── STATS BOXES ──
# Photo stats
photo_stats_box = (
    f"Обычное фото\n"
    f"{'─'*24}\n"
    f"Регионов: {n_photo}\n"
    f"κ_W: [{photo_kw.min():.0f}, {photo_kw.max():.0f}]\n"
    f"κ_W σ: {photo_kw.std():.0f}\n"
    f"R² (яркость): {R2_photo:.3f}\n"
    f"Необъяснено: {(1-R2_photo)*100:.0f}%\n"
    f"Residual σ: {kw_resid_photo.std():.1f}"
)
axes[0, 2].text(0.02, 0.98, photo_stats_box, transform=axes[0, 2].transAxes,
                fontsize=8.5, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#fffde7', alpha=0.9,
                          edgecolor='gray', linewidth=0.5))

jwst_stats_box = (
    f"JWST телескоп\n"
    f"{'─'*24}\n"
    f"Регионов: {len(jwst_stats)}\n"
    f"κ_W: [{jwst_kw.min():.0f}, {jwst_kw.max():.0f}]\n"
    f"κ_W σ: {jwst_kw.std():.0f}\n"
    f"R² (5 фич): {jwst_R2:.3f}\n"
    f"Необъяснено: {(1-jwst_R2)*100:.0f}%\n"
    f"Residual σ: {jwst_kw_resid.std():.1f}"
)
axes[1, 2].text(0.02, 0.98, jwst_stats_box, transform=axes[1, 2].transAxes,
                fontsize=8.5, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.9,
                          edgecolor='gray', linewidth=0.5))

# ── BOTTOM BAR ──
fig.text(0.5, 0.005,
         'κ_W измеряет спектрально-геометрическую структуру, невидимую простым статистикам пикселей.  '
         'Сравнение: обычное фото (верх) vs JWST (низ).  '
         'Обе имеют необъяснённую остаточную структуру, но разного характера и масштаба.',
         ha='center', fontsize=12, style='italic',
         bbox=dict(boxstyle='round', facecolor='#eeeeee', alpha=0.9))

fig.suptitle('Спектральное поле — видимое vs невидимое\n'
             'Обычная камера против космического телескопа JWST',
             fontsize=16, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0.03, 1, 0.94])

fig.savefig('side_by_side_reveal.pdf', dpi=150, bbox_inches='tight')
fig.savefig('side_by_side_reveal.png', dpi=150, bbox_inches='tight')
print('\nSaved side_by_side_reveal.pdf + side_by_side_reveal.png')
print(f'\nSummary:')
print(f'  Photo: {n_photo} regions  κ_W=[{photo_kw.min():.0f},{photo_kw.max():.0f}]  R²={R2_photo:.3f}  resid_σ={kw_resid_photo.std():.1f}')
print(f'  JWST:  {len(jwst_stats)} regions  κ_W=[{jwst_kw.min():.0f},{jwst_kw.max():.0f}]  R²={jwst_R2:.3f}  resid_σ={jwst_kw_resid.std():.1f}')
