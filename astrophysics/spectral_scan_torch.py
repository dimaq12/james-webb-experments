import sys; sys.path.insert(0, '/home/dima/FA/sft_torch')
import numpy as np
from astropy.io import fits
import torch
import sft_torch as sft
import time, os

os.chdir('/home/dima/FA/astrophysics')

hdul = fits.open('jw02736-o001_t001_nircam_clear-f090w_i2d.fits')
data = hdul['SCI'].data.astype(np.float64)
data[np.isnan(data)] = 0.0
H, W = data.shape
print(f'JWST F090W: {W}x{H}  [{data.min():.1f}, {data.max():.1f}]')

n, M = 10, 5
N_OP = n * n
region_size = 80

basis_tensors = []
for d in range(M):
    B = np.zeros((N_OP, N_OP))
    for i in range(N_OP):
        xi, yi = i % n, i // n
        j = yi * n + ((xi + d) % n)
        B[i, j] = 1.0
        B[j, i] = 1.0
    basis_tensors.append(torch.from_numpy(B))
Bs = torch.stack(basis_tensors).to(torch.float64)

def block_is_valid(block):
    return np.count_nonzero(block) > block.size * 0.6

def compute_stats(block):
    pixels = block[::block.shape[0]//n, ::block.shape[1]//n]
    pixels = pixels[:n, :n].ravel()
    pixels = pixels - np.min(pixels) + 0.1

    L = np.zeros((N_OP, N_OP))
    for i in range(N_OP):
        xi, yi = i % n, i // n
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = xi + dx, yi + dy
            if 0 <= nx < n and 0 <= ny < n:
                j = ny * n + nx
                L[i, j] -= 1
                L[i, i] += 1

    A0 = torch.diag(torch.from_numpy(pixels)) + torch.from_numpy(L) * 0.3
    prog = sft.operator(A0.to(torch.float64), Bs).compile("cpu")
    fam = prog.family

    kappa_val = prog.kappa(method="perturbation", n_pert=3, eps=0.1).summary()['value']

    tail = prog.tail(strict=False)
    lam = fam.lam0
    gaps = torch.diff(lam)
    W = fam.W

    decomp = prog.decompose()
    sv = decomp.singular
    kurt = float(((sv - sv.mean())**4).sum() / (len(sv) * sv.std()**4 + 1e-20)) \
        if len(sv) > 1 and sv.std() > 1e-15 else 0.0

    return {
        'complexity': float(fam.complexity),
        'kappa_W': float(kappa_val),
        'kurtosis': kurt,
        'gap_median': float(gaps.median()),
        'gap_min': float(gaps.min()),
        'lam_std': float(lam.std()),
        'lam_range': float(lam[-1] - lam[0]),
        'pixel_std': float(np.std(block)),
        'pixel_mean': float(np.mean(block)),
        'nonzero': np.count_nonzero(block) / block.size,
        'tail_alpha': float(tail.alpha) if tail.alpha else 0.0,
        'tail_class': tail.growth_class,
        'rank': fam.W_rank,
        'cond': float(fam.condition_number()),
    }

stride = 160
stats = []
positions = []
t0 = time.perf_counter()
scan_count = 0

for quadrant_name, i_range, j_range in [
    ('NW', (0, H//2 - region_size), (0, W//2 - region_size)),
    ('NE', (0, H//2 - region_size), (W//2 + region_size, W - region_size)),
    ('SW', (H//2 + region_size, H - region_size), (0, W//2 - region_size)),
    ('SE', (H//2 + region_size, H - region_size), (W//2 + region_size, W - region_size)),
]:
    i0, i1 = i_range
    j0, j1 = j_range
    for i in range(i0, i1, stride):
        for j in range(j0, j1, stride):
            block = data[i:i + region_size, j:j + region_size]
            if block.shape[0] < region_size or block.shape[1] < region_size:
                continue
            if not block_is_valid(block):
                continue
            scan_count += 1
            if scan_count > 200:
                break
            s = compute_stats(block)
            s['i'] = i; s['j'] = j; s['quadrant'] = quadrant_name
            stats.append(s)
            positions.append((i, j))
            if scan_count % 20 == 0:
                print(f'  [{scan_count}] ({i},{j}) κ_W={s["kappa_W"]:.1f} τ_α={s["tail_alpha"]:.3f}')
        if scan_count > 200:
            break
    if scan_count > 200:
        break

elapsed = time.perf_counter() - t0
print(f'\nScanned {len(stats)} valid regions in {elapsed:.0f}s')

metrics = ['kappa_W', 'kurtosis', 'gap_median', 'gap_min', 'lam_std', 'pixel_std', 'tail_alpha']
print(f'\n{"Metric":<14s} {"min":>10s} {"max":>10s} {"mean":>10s} {"std":>10s} {"range%":>8s}')
for k in metrics:
    vals = np.array([s[k] for s in stats])
    rng_pct = 100 * (np.max(vals) - np.min(vals)) / np.mean(vals) if np.mean(vals) > 1e-10 else 0
    print(f'{k:<14s} {np.min(vals):>10.4f} {np.max(vals):>10.4f} '
          f'{np.mean(vals):>10.4f} {np.std(vals):>10.4f} {rng_pct:>7.1f}%')

kw_vals = np.array([s['kappa_W'] for s in stats])
kw_z = (kw_vals - np.mean(kw_vals)) / np.std(kw_vals)
print(f'\n─── κ_W ANOMALIES (|z| > 2σ) ───')
for idx in np.argsort(-np.abs(kw_z)):
    if abs(kw_z[idx]) < 2.0:
        break
    s = stats[idx]
    print(f'  [{s["i"]:>4d},{s["j"]:>4d}] {s["quadrant"]} '
          f'κ_W={s["kappa_W"]:.1f} z={kw_z[idx]:+.1f}σ '
          f'α={s["tail_alpha"]:.3f} {s["tail_class"][:20]}')

results = {
    'stats': stats,
    'positions': positions,
    'image_shape': (H, W),
    'region_size': region_size,
    'stride': stride,
    'engine': 'sft_torch',
}
np.save('spectral_scan_torch_results.npy', results, allow_pickle=True)
print(f'\nSaved spectral_scan_torch_results.npy ({len(stats)} regions)')
