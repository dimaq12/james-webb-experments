import sys; sys.path.insert(0, '/home/dima/FA/sft_torch')
import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import torch, sft_torch as sft
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, os; os.chdir('/home/dima/FA/astrophysics')

n, M = 10, 5; N = n * n; RSZ = 80; STRIDE = 80

basis_tensors = []
for d in range(M):
    B = np.zeros((N, N))
    for i in range(N):
        xi, yi = i % n, i // n
        j = yi * n + ((xi + d) % n)
        B[i, j] = 1.0; B[j, i] = 1.0
    basis_tensors.append(torch.from_numpy(B))
Bs = torch.stack(basis_tensors).to(torch.float64)

def compute_omega_block(block):
    px = block[::RSZ//n, ::RSZ//n][:n, :n].ravel()
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
    lam = fam.lam0
    om = tail.omega.cpu().numpy()
    om_norm = float(np.linalg.norm(om))
    # angular representation: normalize to unit vector
    om_unit = om / (om_norm + 1e-15)
    return {
        'kw': float(kw),
        'alpha': float(tail.alpha if tail.alpha else 0),
        'om_norm': om_norm,
        'om_vec': om,
        'om_unit': om_unit,
        'lam_min': float(lam[0]),
        'lam_max': float(lam[-1]),
        'pixel_mean': float(block.mean()),
        'pixel_std': float(block.std()),
    }

# ═══ LOAD SMACS F090W ═══
print("Loading SMACS F090W...")
hdul = fits.open('jw02736-o001_t001_nircam_clear-f090w_i2d.fits')
data = hdul['SCI'].data.astype(np.float64)
data[np.isnan(data)] = 0.0
H, W = data.shape
print(f'  {W}x{H}')

print('Scanning with Omega vectors...')
t0 = time.perf_counter()
stats = []

for qname, i_range, j_range in [
    ('NW', (0, H//2 - RSZ), (0, W//2 - RSZ)),
    ('NE', (0, H//2 - RSZ), (W//2 + RSZ, W - RSZ)),
    ('SW', (H//2 + RSZ, H - RSZ), (0, W//2 - RSZ)),
    ('SE', (H//2 + RSZ, H - RSZ), (W//2 + RSZ, W - RSZ)),
]:
    i0, i1 = i_range; j0, j1 = j_range
    for i in range(i0, i1, STRIDE):
        if len(stats) >= 250: break
        for j in range(j0, j1, STRIDE):
            if len(stats) >= 250: break
            block = data[i:i+RSZ, j:j+RSZ]
            if block.shape[0] < RSZ or block.shape[1] < RSZ: continue
            if np.count_nonzero(block) < block.size * 0.5: continue
            s = compute_omega_block(block)
            s['i'] = i; s['j'] = j
            s['ci'] = i + RSZ//2; s['cj'] = j + RSZ//2
            stats.append(s)
            if len(stats) % 50 == 0:
                print(f'  [{len(stats)}] ({i},{j}) kw={s["kw"]:.0f} ||Ω||={s["om_norm"]:.4f}')
        if len(stats) >= 250: break
    if len(stats) >= 250: break

n_reg = len(stats)
dt = time.perf_counter() - t0
print(f'  {n_reg} regions in {dt:.0f}s')

kw = np.array([s['kw'] for s in stats])
al = np.array([s['alpha'] for s in stats])
om_norm = np.array([s['om_norm'] for s in stats])
ci = np.array([s['ci'] for s in stats])
cj = np.array([s['cj'] for s in stats])
om_units = np.array([s['om_unit'] for s in stats])  # (n, 4)

# Omega angular spread per region relative to mean direction
om_mean_dir = om_units.mean(axis=0)
om_mean_dir /= np.linalg.norm(om_mean_dir)
angular_dev = 1.0 - np.abs(om_units @ om_mean_dir)  # deviation from mean direction

cv_al = al.std()/al.mean()
cv_om = om_norm.std()/om_norm.mean()

print(f'\ncv(α) = {cv_al:.4f}  cv(||Ω||) = {cv_om:.4f}')
print(f'κ_W:  [{kw.min():.0f}, {kw.max():.0f}] μ={kw.mean():.0f}±{kw.std():.0f}')
print(f'||Ω||: [{om_norm.min():.4f}, {om_norm.max():.4f}] μ={om_norm.mean():.4f}±{om_norm.std():.4f}')

# ═══ BUILD DENSE MAPS ═══
def build_map(vals):
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    dense = griddata((ci, cj), vals, (grid_y, grid_x), method='linear', fill_value=np.nan)
    mask = ~np.isnan(dense)
    dense[~mask] = 0
    dense = gaussian_filter(dense, sigma=2.5)
    dense[~mask] = np.nan
    return dense

print('Building dense maps...')
kw_map = build_map(kw)
om_norm_map = build_map(om_norm)
angular_map = build_map(angular_dev)
alpha_map = build_map(al)

# ═══ FIGURE: 2 rows × 4 cols ═══
# Row 1: image + κ_W + ||Ω|| + α
# Row 2: image + κ_W overlay + ||Ω|| overlay + Ω angular deviation

fig, axes = plt.subplots(2, 4, figsize=(26, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.18)

# crop to a good display region (NW quadrant has clearest data)
crop_h, crop_w = 1200, 2000
i0_d, j0_d = 30, 30

crop = data[i0_d:i0_d+crop_h, j0_d:j0_d+crop_w]
crop_log = np.log10(crop - crop.min() + 0.01)

kw_disp = kw_map[i0_d:i0_d+crop_h, j0_d:j0_d+crop_w]
om_disp = om_norm_map[i0_d:i0_d+crop_h, j0_d:j0_d+crop_w]
ang_disp = angular_map[i0_d:i0_d+crop_h, j0_d:j0_d+crop_w]
al_disp = alpha_map[i0_d:i0_d+crop_h, j0_d:j0_d+crop_w]

# ── ROW 1: RAW MAPS ──
ax = axes[0, 0]
ax.imshow(crop_log, cmap='inferno', origin='lower',
          vmin=np.percentile(crop_log, 5), vmax=np.percentile(crop_log, 99))
ax.set_title('SMACS 0723 F090W\n(исходное изображение)', fontsize=11, fontweight='bold')

ax = axes[0, 1]
im = ax.imshow(kw_disp, cmap='plasma', origin='lower', interpolation='bilinear')
ax.set_title(f'κ_W спектральная кривизна\n[{kw.min():.0f}..{kw.max():.0f}]', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.7, label='κ_W')

ax = axes[0, 2]
im = ax.imshow(om_disp, cmap='viridis', origin='lower', interpolation='bilinear')
ax.set_title(f'||Ω|| норма омега-вектора\n[{om_norm.min():.3f}..{om_norm.max():.3f}]', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.7, label='||Ω||')

ax = axes[0, 3]
im = ax.imshow(al_disp, cmap='coolwarm', origin='lower', interpolation='bilinear')
ax.set_title(f'Tail α\n[{al.min():.3f}..{al.max():.3f}]', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.7, label='α')

# ── ROW 2: OVERLAY ON IMAGE ──
ax = axes[1, 0]
ax.imshow(crop_log, cmap='inferno', origin='lower',
          vmin=np.percentile(crop_log, 5), vmax=np.percentile(crop_log, 99))
ax.set_title('Изображение (повтор)', fontsize=11, fontweight='bold')

ax = axes[1, 1]
ax.imshow(crop_log, cmap='gray', origin='lower', alpha=0.3,
          vmin=np.percentile(crop_log, 10), vmax=np.percentile(crop_log, 95))
im = ax.imshow(kw_disp, cmap='plasma', origin='lower', alpha=0.7, interpolation='bilinear')
ax.set_title(f'κ_W наложенное на кадр\ncv(κ)={kw.std()/kw.mean():.4f}', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.7, label='κ_W')

ax = axes[1, 2]
ax.imshow(crop_log, cmap='gray', origin='lower', alpha=0.3,
          vmin=np.percentile(crop_log, 10), vmax=np.percentile(crop_log, 95))
im = ax.imshow(om_disp, cmap='viridis', origin='lower', alpha=0.7, interpolation='bilinear')
ax.set_title(f'||Ω|| наложенное на кадр\ncv(||Ω||)={cv_om:.4f}', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.7, label='||Ω||')

ax = axes[1, 3]
ax.imshow(crop_log, cmap='gray', origin='lower', alpha=0.3,
          vmin=np.percentile(crop_log, 10), vmax=np.percentile(crop_log, 95))
im = ax.imshow(ang_disp, cmap='RdBu_r', origin='lower', alpha=0.7, interpolation='bilinear')
ax.set_title(f'Ω angular deviation\nот среднего направления', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.7, label='1-cos(θ)')

# stats box
info = (
    f"SMACS 0723 F090W — Omega field\n"
    f"{'─'*28}\n"
    f"Регионов: {n_reg}\n"
    f"N={N} M={M} RSZ={RSZ}\n\n"
    f"κ_W:  [{kw.min():.0f}..{kw.max():.0f}]\n"
    f"α:    [{al.min():.3f}..{al.max():.3f}]\n"
    f"||Ω||: [{om_norm.min():.3f}..{om_norm.max():.3f}]\n\n"
    f"cv(α)     = {cv_al:.4f}\n"
    f"cv(||Ω||)  = {cv_om:.4f}\n"
    f"cv(κ_W)   = {kw.std()/kw.mean():.4f}\n"
)
axes[1, 3].text(0.02, 0.22, info, transform=axes[1, 3].transAxes, fontsize=7.5,
                verticalalignment='bottom', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#fffff5', alpha=0.92,
                          edgecolor='gray', linewidth=0.5))

fig.suptitle('Ω (Omega) Spectral Field — SMACS 0723 JWST NIRCam F090W\n'
             'κ_W + ||Ω|| + Tail α — sft_torch',
             fontsize=15, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('smacs_omega_field.png', dpi=150, bbox_inches='tight')
fig.savefig('smacs_omega_field.pdf', dpi=150, bbox_inches='tight')
print('\nSaved smacs_omega_field.png + smacs_omega_field.pdf')
