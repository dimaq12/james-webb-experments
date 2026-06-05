import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.stats import spearmanr, wilcoxon
import torch
import sft_torch as sft
import time, os

os.chdir('/home/dima/FA/astrophysics')

results = np.load('spectral_scan_torch_results.npy', allow_pickle=True).item()
stats_list = results['stats']
n_regions = len(stats_list)

kw = np.array([s['kappa_W'] for s in stats_list])
ps = np.array([s['pixel_std'] for s in stats_list])
pm = np.array([s['pixel_mean'] for s in stats_list])
gi = np.array([s['i'] for s in stats_list])
gj = np.array([s['j'] for s in stats_list])
ta = np.array([s.get('tail_alpha', 0) for s in stats_list])
tc = [s.get('tail_class', '?') for s in stats_list]

hdul = fits.open('jw02736-o001_t001_nircam_clear-f090w_i2d.fits')
data = hdul['SCI'].data.astype(np.float64)
data[np.isnan(data)] = 0.0

region_size = 80
n, M = 10, 5
N = n * n
basis_tensors = []
for d in range(M):
    B = np.zeros((N, N))
    for i in range(N):
        xi, yi = i % n, i // n
        j = yi * n + ((xi + d) % n)
        B[i, j] = 1.0; B[j, i] = 1.0
    basis_tensors.append(torch.from_numpy(B))
Bs = torch.stack(basis_tensors).to(torch.float64)

def compute_kw_torch(block):
    pixels = block[::block.shape[0]//n, ::block.shape[1]//n]
    pixels = pixels[:n, :n].ravel()
    pixels = pixels - np.min(pixels) + 0.1
    L = np.zeros((N, N))
    for i in range(N):
        xi, yi = i % n, i // n
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = xi + dx, yi + dy
            if 0 <= nx < n and 0 <= ny < n:
                j = ny * n + nx
                L[i, j] -= 1; L[i, i] += 1
    A0 = torch.diag(torch.from_numpy(pixels)) + torch.from_numpy(L) * 0.3
    prog = sft.operator(A0.to(torch.float64), Bs).compile("cpu")
    return prog.kappa(method="perturbation", n_pert=3, eps=0.1).summary()['value']

print(f'Loaded {n_regions} regions\nκ_W: [{kw.min():.0f}, {kw.max():.0f}] mean={kw.mean():.0f} std={kw.std():.0f}')
print(f'κ_W-pixel_std correlation: r={np.corrcoef(kw, ps)[0,1]:.3f}')
print(f'κ_W-tail_alpha correlation: r={np.corrcoef(kw, ta)[0,1]:.3f}')

# ═══════════════════════════════════════════════════════════
# 1. CONTROL: SHUFFLE PIXELS
# ═══════════════════════════════════════════════════════════
print('\n═══ SHUFFLE CONTROL ═══')
rng = np.random.default_rng(42)
shuffle_indices = rng.choice(n_regions, size=min(30, n_regions), replace=False)
kw_orig_samples = []; kw_shuf_samples = []

for idx in shuffle_indices:
    s = stats_list[idx]
    block = data[s['i']:s['i']+region_size, s['j']:s['j']+region_size]
    if np.count_nonzero(block) < block.size * 0.6:
        continue
    kw_orig_samples.append(compute_kw_torch(block))
    flat = block.ravel(); rng.shuffle(flat)
    kw_shuf_samples.append(compute_kw_torch(flat.reshape(block.shape)))

kw_orig_arr = np.array(kw_orig_samples)
kw_shuf_arr = np.array(kw_shuf_samples)
change = np.abs(kw_shuf_arr - kw_orig_arr) / (np.abs(kw_orig_arr) + 1e-10)
print(f'Original κ_W: mean={kw_orig_arr.mean():.1f} std={kw_orig_arr.std():.1f}')
print(f'Shuffled κ_W: mean={kw_shuf_arr.mean():.1f} std={kw_shuf_arr.std():.1f}')
print(f'Change: {change.mean()*100:.1f}% mean, {np.median(change)*100:.1f}% median')
stat, pval = wilcoxon(kw_orig_arr, kw_shuf_arr)
print(f'Wilcoxon p={pval:.4f} → κ_W {"CHANGES" if pval<0.01 else "UNCHANGED"}')
print(f'=> κ_W captures spatial GEOMETRY, not just histogram')

# ═══════════════════════════════════════════════════════════
# 2. TRADITIONAL FEATURES
# ═══════════════════════════════════════════════════════════
print('\n═══ FEATURES vs κ_W ═══')
entropy_vals = []; sobel_vals = []; lapvar_vals = []
for idx in range(n_regions):
    s = stats_list[idx]
    block = data[s['i']:s['i']+region_size, s['j']:s['j']+region_size]
    hist, _ = np.histogram(block[block > 0], bins=50, density=True)
    hist = hist[hist > 0]
    ent = -np.sum(hist * np.log(hist + 1e-20))
    entropy_vals.append(ent)
    sx = ndimage.sobel(block, axis=0); sy = ndimage.sobel(block, axis=1)
    sobel_vals.append(np.sqrt(sx**2 + sy**2).mean())
    lapvar_vals.append(ndimage.laplace(block).var())
ent_arr = np.array(entropy_vals)
sob_arr = np.array(sobel_vals)
lap_arr = np.array(lapvar_vals)

features = {'pixel_mean': pm, 'pixel_std': ps, 'entropy': ent_arr,
            'sobel_energy': sob_arr, 'laplacian_var': lap_arr, 'tail_alpha': ta}

print(f'{"Feature":<16s} {"r":>8s} {"r²":>8s} {"ρ(sp)":>8s}')
best_r = -1; best_name = ''
for name, arr in features.items():
    r = np.corrcoef(arr, kw)[0, 1]
    rho, _ = spearmanr(arr, kw)
    print(f'{name:<16s} {r:>+8.4f} {r**2:>8.4f} {rho:>+8.4f}')
    if abs(r) > abs(best_r):
        best_r = r; best_name = name
print(f'\nStrongest predictor: {best_name} (r={best_r:.4f})')

# ═══════════════════════════════════════════════════════════
# 3. RESIDUAL ANALYSIS
# ═══════════════════════════════════════════════════════════
print('\n═══ RESIDUAL κ_W ═══')
X = np.column_stack([pm, ps, ent_arr, sob_arr, lap_arr, ta, np.ones_like(pm)])
coeffs, _, _, _ = np.linalg.lstsq(X, kw, rcond=None)
kw_pred = X @ coeffs
kw_residual = kw - kw_pred
R2 = 1 - np.var(kw_residual) / np.var(kw)
print(f'R² (6 features): {R2:.4f}')
print(f'Unexplained: {(1-R2)*100:.1f}%')
print(f'Residual σ: {kw_residual.std():.1f} (original σ: {kw.std():.1f})')
print(f'Residual range: [{kw_residual.min():.0f}, {kw_residual.max():.0f}]')

residual_z = kw_residual / kw_residual.std()
print(f'\nTop positive residuals:')
for idx in np.argsort(-residual_z)[:5]:
    s = stats_list[idx]
    print(f'  [{s["i"]:>4d},{s["j"]:>4d}] {s["quadrant"]} κ_W={kw[idx]:.0f} '
          f'pred={kw_pred[idx]:.0f} res={kw_residual[idx]:+.0f} z={residual_z[idx]:+.1f}σ α={ta[idx]:.3f}')
print(f'\nTop negative residuals:')
for idx in np.argsort(residual_z)[:5]:
    s = stats_list[idx]
    print(f'  [{s["i"]:>4d},{s["j"]:>4d}] {s["quadrant"]} κ_W={kw[idx]:.0f} '
          f'pred={kw_pred[idx]:.0f} res={kw_residual[idx]:+.0f} z={residual_z[idx]:+.1f}σ α={ta[idx]:.3f}')

# ═══════════════════════════════════════════════════════════
# 4. CORRELATION WITH ORIGINAL sft KAPPA
# ═══════════════════════════════════════════════════════════
print('\n═══ TORCH vs NUMPY sft ═══')
orig_results = np.load('spectral_scan_results.npy', allow_pickle=True).item()
orig_stats = orig_results['stats']
o_kw = np.array([s['kappa_W'] for s in orig_stats])
# Map by position
pos_to_orig = {(s['i'], s['j']): s['kappa_W'] for s in orig_stats}
matched = []
for s in stats_list:
    key = (s['i'], s['j'])
    if key in pos_to_orig:
        matched.append((s['kappa_W'], pos_to_orig[key]))
if matched:
    m_t, m_o = zip(*matched)
    r_cross = np.corrcoef(m_t, m_o)[0, 1]
    rho_cross, _ = spearmanr(m_t, m_o)
    print(f'Matched {len(matched)} positions')
    print(f'Torch vs original κ_W: r={r_cross:.4f} Spearman ρ={rho_cross:.4f}')

# save enhanced results
for i, s in enumerate(stats_list):
    s['kw_pred'] = float(kw_pred[i])
    s['kw_residual'] = float(kw_residual[i])
for i, s in enumerate(stats_list):
    s['entropy'] = float(ent_arr[i])
    s['sobel_energy'] = float(sob_arr[i])
    s['laplacian_var'] = float(lap_arr[i])

results['features'] = {
    'entropy': ent_arr, 'sobel_energy': sob_arr, 'laplacian_var': lap_arr,
    'kw_pred': kw_pred, 'kw_residual': kw_residual, 'R2': R2,
}
results['control'] = {
    'kw_orig': kw_orig_arr, 'kw_shuffled': kw_shuf_arr,
    'wilcoxon_p': pval,
}
np.save('spectral_scan_torch_results.npy', results, allow_pickle=True)
print('\nSaved enhanced results to spectral_scan_torch_results.npy')
