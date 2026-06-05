import numpy as np
from PIL import Image
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import torch, sft_torch as sft
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, os; os.chdir('/home/dima/FA/astrophysics')

# ═══ PARAMETERS ═══
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

def compute_block(block, rng_seed):
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
    kw = prog.kappa(method="perturbation", n_pert=3, eps=0.1).summary()['value']
    return {
        'kw': float(kw),
        'alpha': float(tail.alpha if tail.alpha else 0),
        'class': tail.growth_class,
        'lam0_min': float(fam.lam0[0]),
        'lam0_max': float(fam.lam0[-1]),
        'rank': fam.W_rank,
        'cond': float(fam.condition_number()),
    }

# ═══ LOAD + SCAN ═══
print("Loading DSCF1809.JPG...")
img = Image.open('imgs/DSCF1809.JPG').convert('L')
# downsample 4x for speed
img = img.resize((img.width//4, img.height//4), Image.LANCZOS)
arr = np.array(img, dtype=np.float64) / 255.0
H, W = arr.shape
print(f'  {W}x{H}  [{arr.min():.3f}, {arr.max():.3f}]')

print('Scanning regions...')
t0 = time.perf_counter()
stats = []
for i in range(0, H - region_size, stride):
    if len(stats) >= 300: break
    for j in range(0, W - region_size, stride):
        if len(stats) >= 300: break
        block = arr[i:i+region_size, j:j+region_size]
        if block.max() - block.min() < 0.001: continue
        s = compute_block(block, len(stats))
        s['i'] = i; s['j'] = j
        s['center_i'] = i + region_size//2; s['center_j'] = j + region_size//2
        s['pixel_mean'] = float(block.mean())
        s['pixel_std'] = float(block.std())
        stats.append(s)
        if len(stats) % 30 == 0:
            print(f'  [{len(stats)}] ({i},{j}) κ_W={s["kw"]:.0f} α={s["alpha"]:.3f}')

elapsed = time.perf_counter() - t0
kw = np.array([s['kw'] for s in stats])
alpha = np.array([s['alpha'] for s in stats])
pm = np.array([s['pixel_mean'] for s in stats])
ps = np.array([s['pixel_std'] for s in stats])
print(f'\nScanned {len(stats)} regions in {elapsed:.0f}s')
print(f'κ_W: [{kw.min():.0f}, {kw.max():.0f}] μ={kw.mean():.0f}±{kw.std():.0f}')
print(f'α: [{alpha.min():.3f}, {alpha.max():.3f}] μ={alpha.mean():.3f}±{alpha.std():.3f}')

# ═══ REGRESS κ_W ON BRIGHTNESS ═══
X = np.column_stack([pm, ps, np.ones_like(pm)])
coeffs, _, _, _ = np.linalg.lstsq(X, kw, rcond=None)
kw_pred = X @ coeffs
kw_resid = kw - kw_pred
R2 = 1 - np.var(kw_resid) / np.var(kw)
print(f'R² (яркость+контраст): {R2:.3f}  необъяснено: {(1-R2)*100:.0f}%  residual σ: {kw_resid.std():.1f}')

# ═══ BUILD DENSE MAPS ═══
def build_dense(stats_arr, values, H, W):
    centers_i = np.array([s['center_i'] for s in stats_arr])
    centers_j = np.array([s['center_j'] for s in stats_arr])
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    dense = griddata((centers_i, centers_j), values, (grid_y, grid_x),
                     method='linear', fill_value=np.nan)
    mask = ~np.isnan(dense)
    dense[~mask] = 0
    dense = gaussian_filter(dense, sigma=2.5)
    dense[~mask] = np.nan
    return dense

kw_map = build_dense(stats, kw, H, W)
alpha_map = build_dense(stats, alpha, H, W)
resid_map = build_dense(stats, kw_resid, H, W)

# ═══ LOAD JWST FOR COMPARISON ═══
jwst_res = np.load('spectral_scan_torch_results.npy', allow_pickle=True).item()
jwst_kw = np.array([s['kappa_W'] for s in jwst_res['stats']])
jwst_alpha = np.array([s['tail_alpha'] for s in jwst_res['stats']])
jwst_resid = jwst_res['features']['kw_residual']
jwst_R2 = jwst_res['features']['R2']

# JWST crops
hdul = fits.open('jw02736-o001_t001_nircam_clear-f090w_i2d.fits')
jwst_data = hdul['SCI'].data.astype(np.float64)
jwst_data[np.isnan(jwst_data)] = 0.0
jwst_crop = jwst_data[:1100, :1100]
jwst_stats = jwst_res['stats']
jwst_gi = np.array([s['i'] for s in jwst_stats])
jwst_gj = np.array([s['j'] for s in jwst_stats])

def build_jwst_dense(values):
    return build_dense(
        [{'center_i': jwst_gi[i] + 40, 'center_j': jwst_gj[i] + 40} for i in range(len(jwst_stats))],
        values, jwst_data.shape[0], jwst_data.shape[1])[:1100, :1100]

jwst_kw_map = build_jwst_dense(jwst_kw)
jwst_resid_map = build_jwst_dense(jwst_resid)

# ═══ FIGURE ═══
fig, axes = plt.subplots(2, 3, figsize=(22, 13))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# ── ROW 1: DSCF1809 ──
ax = axes[0, 0]
ax.imshow(arr, cmap='gray', origin='lower')
ax.set_title('A: DSCF1809 — обычное фото\n(то, что видит камера)', fontsize=13, fontweight='bold')
ax.set_xlabel('pixels'); ax.set_ylabel('pixels')

ax = axes[0, 1]
ax.imshow(arr, cmap='gray', origin='lower', alpha=0.25)
im = ax.imshow(kw_map, cmap='plasma', origin='lower', alpha=0.75, interpolation='bilinear')
ax.set_title(f'B: κ_W — спектральная кривизна\n[{kw.min():.0f}..{kw.max():.0f}]  R²={R2:.2f}',
             fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.75); cbar.set_label('κ_W', fontsize=9)

ax = axes[0, 2]
ax.imshow(arr, cmap='gray', origin='lower',
          vmin=np.percentile(arr, 5), vmax=np.percentile(arr, 95), alpha=0.5)
vlim = max(abs(np.nanmin(resid_map)), abs(np.nanmax(resid_map))) if np.any(~np.isnan(resid_map)) else 40
im = ax.imshow(resid_map, cmap='RdBu_r', origin='lower', alpha=0.65,
               interpolation='bilinear', vmin=-vlim, vmax=vlim)
ax.set_title(f'C: Невидимое спектральное поле\n'
             f'(κ_W residual) σ={kw_resid.std():.1f}  необъяснено={(1-R2)*100:.0f}%',
             fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.75); cbar.set_label('Residual κ_W', fontsize=9)

# stats box
photo_stats = (
    f"DSCF1809\n{'─'*20}\n"
    f"Регионов: {len(stats)}\n"
    f"κ_W: [{kw.min():.0f}, {kw.max():.0f}]\n"
    f"  μ={kw.mean():.0f} σ={kw.std():.0f}\n"
    f"α: [{alpha.min():.3f}, {alpha.max():.3f}]\n"
    f"  μ={alpha.mean():.3f} σ={alpha.std():.3f}\n"
    f"R² (яркость): {R2:.3f}\n"
    f"Необъяснено: {(1-R2)*100:.0f}%\n"
    f"Residual σ: {kw_resid.std():.1f}"
)
axes[0, 2].text(0.02, 0.98, photo_stats, transform=axes[0, 2].transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#fffde7', alpha=0.9,
                          edgecolor='gray', linewidth=0.5))

# ── ROW 2: JWST ──
ax = axes[1, 0]
log_jwst = np.log10(jwst_crop - jwst_crop.min() + 0.01)
valid = jwst_crop[jwst_crop > 0]
vmin_j, vmax_j = np.percentile(valid, 1), np.percentile(valid, 99.5)
ax.imshow(jwst_crop, cmap='inferno', origin='lower', vmin=vmin_j, vmax=vmax_j)
ax.set_title('D: JWST NIRCam F090W\n(то, что видит телескоп)', fontsize=13, fontweight='bold')
ax.set_xlabel('pixels'); ax.set_ylabel('pixels')

ax = axes[1, 1]
ax.imshow(log_jwst, cmap='gray', origin='lower', alpha=0.3,
          vmin=np.percentile(log_jwst, 20), vmax=np.percentile(log_jwst, 95))
im = ax.imshow(jwst_kw_map, cmap='plasma', origin='lower', alpha=0.7, interpolation='bilinear')
ax.set_title(f'E: JWST κ_W — спектральная кривизна\n'
             f'[{jwst_kw.min():.0f}..{jwst_kw.max():.0f}]  R²={jwst_R2:.2f}',
             fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.75); cbar.set_label('κ_W', fontsize=9)

ax = axes[1, 2]
ax.imshow(log_jwst, cmap='gray', origin='lower',
          vmin=np.percentile(log_jwst, 20), vmax=np.percentile(log_jwst, 95), alpha=0.5)
im = ax.imshow(jwst_resid_map, cmap='RdBu_r', origin='lower', alpha=0.65,
               interpolation='bilinear', vmin=-60, vmax=50)
ax.set_title(f'F: Невидимое спектральное поле JWST\n'
             f'(κ_W residual) σ={jwst_resid.std():.1f}  необъяснено={(1-jwst_R2)*100:.0f}%',
             fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.75); cbar.set_label('Residual κ_W', fontsize=9)

jwst_stats_box = (
    f"JWST SMACS 0723\n{'─'*20}\n"
    f"Регионов: {len(jwst_stats)}\n"
    f"κ_W: [{jwst_kw.min():.0f}, {jwst_kw.max():.0f}]\n"
    f"  μ={jwst_kw.mean():.0f} σ={jwst_kw.std():.0f}\n"
    f"α: [{jwst_alpha.min():.3f}, {jwst_alpha.max():.3f}]\n"
    f"  μ={jwst_alpha.mean():.3f} σ={jwst_alpha.std():.3f}\n"
    f"R² (5 фич): {jwst_R2:.3f}\n"
    f"Необъяснено: {(1-jwst_R2)*100:.0f}%\n"
    f"Residual σ: {jwst_resid.std():.1f}"
)
axes[1, 2].text(0.02, 0.98, jwst_stats_box, transform=axes[1, 2].transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.9,
                          edgecolor='gray', linewidth=0.5))

# ── ANNOTATIONS ON RESIDUAL MAPS ──
# Photo: mark regions
resid_clean = resid_map.copy()
resid_clean[np.isnan(resid_clean)] = 0
flat_sort = np.argsort(-resid_clean.ravel())

for rank, (label, color, y_offset) in enumerate([
    ('Сложнее\nожидаемого', 'darkred', 80),
    ('Проще\nожидаемого', 'darkblue', -80),
]):
    for offset in range(rank * 400, (rank+1) * 400):
        idx = flat_sort[offset]
        cy, cx = idx // W, idx % W
        val = resid_map[cy, cx]
        if not np.isnan(val) and 30 < cy < H-30 and 30 < cx < W-30:
            break
    axes[0, 2].annotate(f'{label}', xy=(cx, cy),
                        fontsize=10, fontweight='bold', color=color,
                        xytext=(cx + 200, cy + y_offset),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.5),
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), ha='center')

# JWST annotations
axes[1, 2].annotate('Проще', xy=(80, 80), fontsize=10, fontweight='bold', color='darkblue',
                    xytext=(250, 30), arrowprops=dict(arrowstyle='->', color='darkblue', lw=2.5),
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), ha='center')
axes[1, 2].annotate('Сложнее', xy=(620, 580), fontsize=10, fontweight='bold', color='darkred',
                    xytext=(850, 700), arrowprops=dict(arrowstyle='->', color='darkred', lw=2.5),
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), ha='center')

# ── BOTTOM ──
fig.text(0.5, 0.005,
         'κ_W = спектральная кривизна операторного семейства.  '
         'Residual = то, что остаётся после регрессии κ_W на яркость и контраст — скрытая спектрально-геометрическая структура.  '
         'DSCF1809 (фото) vs JWST (телескоп): разный масштаб и характер остаточного поля.',
         ha='center', fontsize=11, style='italic',
         bbox=dict(boxstyle='round', facecolor='#eeeeee', alpha=0.9))

fig.suptitle('Спектральное поле DSCF1809 vs JWST\nНевидимая структура поверх кадра',
             fontsize=16, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.savefig('dscf1809_reveal.pdf', dpi=150, bbox_inches='tight')
fig.savefig('dscf1809_reveal.png', dpi=150, bbox_inches='tight')
print(f'\nSaved dscf1809_reveal.pdf + dscf1809_reveal.png')
