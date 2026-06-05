import numpy as np
from scipy.stats import spearmanr, ks_2samp, mannwhitneyu
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os; os.chdir('/home/dima/FA/astrophysics')

# ═══ LOAD ═══
smacs = np.load('spectral_scan_torch_results.npy', allow_pickle=True).item()
hudf_data = np.load('hst_jwst_torch_results.npz', allow_pickle=True)

sm_kw = np.array([s['kappa_W'] for s in smacs['stats']])
sm_al = np.array([s['tail_alpha'] for s in smacs['stats']])
sm_ps = np.array([s['pixel_std'] for s in smacs['stats']])
sm_pm = np.array([s['pixel_mean'] for s in smacs['stats']])
sm_gi = np.array([s['i'] for s in smacs['stats']])
sm_gj = np.array([s['j'] for s in smacs['stats']])

jw_label = 'HUDF JWST F150W'
hu_kw = np.array(hudf_data['kw_results'].item()[jw_label])
hu_al = np.array(hudf_data['alpha_results'].item()[jw_label])
hu_ps = np.array(hudf_data['ps_results'].item()[jw_label])
hu_pm = np.array(hudf_data['pm_results'].item()[jw_label])
hu_pos = hudf_data['pos_results'].item()[jw_label]
hu_gi = np.array([p[0] for p in hu_pos])
hu_gj = np.array([p[1] for p in hu_pos])
hu_lm = hudf_data['lam0m_results'].item()[jw_label]

hst_label = 'HST WFC3 F160W'
ht_kw = np.array(hudf_data['kw_results'].item()[hst_label])
ht_al = np.array(hudf_data['alpha_results'].item()[hst_label])
ht_ps = np.array(hudf_data['ps_results'].item()[hst_label])
ht_pm = np.array(hudf_data['pm_results'].item()[hst_label])
ht_pos = hudf_data['pos_results'].item()[hst_label]
ht_gi = np.array([p[0] for p in ht_pos])
ht_gj = np.array([p[1] for p in ht_pos])

# DSCF1809 from earlier scan — reuse iterated_residuals.py data
# We computed it in iterated_residuals_viz.py; just quick approximate here
from PIL import Image
import sys; sys.path.insert(0, '/home/dima/FA/sft_torch')
import torch, sft_torch as sft

# quick helper
def regress(y, X):
    c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ c; resid = y - pred
    R2 = 1 - np.var(resid)/np.var(y) if np.var(y) > 0 else 0
    return pred, resid, R2

def iterated_regression(kw, al, ps, pm, gi, gj, lmins, lmaxs, lstds):
    gin = (gi - gi.mean())/gi.std()
    gjn = (gj - gj.mean())/gj.std()

    _, resid0, R2_0 = regress(kw, np.column_stack([pm, ps, np.ones_like(pm)]))
    _, resid1, R2_1 = regress(resid0, np.column_stack([
        al, lstds, lmins, lmaxs, gin, gjn, gin**2, gjn**2, gin*gjn, np.ones_like(al)]))
    cum = R2_0 + (1-R2_0)*R2_1
    return R2_0, R2_1, cum, resid0, resid1

# SMACS
sm_lmins = np.array([s['lam0_min'] if 'lam0_min' in s else sm_kw.mean() for s in smacs['stats']])
sm_lmaxs = np.array([s['lam0_max'] if 'lam0_max' in s else sm_kw.mean() for s in smacs['stats']])
sm_lstds = np.array([s['lam_std'] if 'lam_std' in s else sm_kw.std() for s in smacs['stats']])
r = iterated_regression(sm_kw, sm_al, sm_ps, sm_pm, sm_gi, sm_gj, sm_lmins, sm_lmaxs, sm_lstds)
sm_R0, sm_R1, sm_cum, sm_r0, sm_r1 = r

# HUDF JWST
hu_lmins = np.array([l[0] for l in hu_lm])
hu_lmaxs = np.array([l[1] for l in hu_lm])
hu_lstds = np.array([l[2] for l in hu_lm])
r = iterated_regression(hu_kw, hu_al, hu_ps, hu_pm, hu_gi, hu_gj, hu_lmins, hu_lmaxs, hu_lstds)
hu_R0, hu_R1, hu_cum, hu_r0, hu_r1 = r

# HST WFC3
ht_lmins = np.array([l[0] for l in hudf_data['lam0m_results'].item()[hst_label]])
ht_lmaxs = np.array([l[1] for l in hudf_data['lam0m_results'].item()[hst_label]])
ht_lstds = np.array([l[2] for l in hudf_data['lam0m_results'].item()[hst_label]])
r = iterated_regression(ht_kw, ht_al, ht_ps, ht_pm, ht_gi, ht_gj, ht_lmins, ht_lmaxs, ht_lstds)
ht_R0, ht_R1, ht_cum, ht_r0, ht_r1 = r

# ═══ PRINT TABLE ═══
print("=" * 100)
print("FIELD TYPE vs SPECTRAL METRICS — JWST (identical setup: 10×10 grid, M=5, RSZ=80)")
print("=" * 100)
print()
print(f'{"Field":<22s} {"N":>5s} {"κ_W μ±σ":>14s} {"α μ±σ":>18s} {"cv(α)":>8s} {"R²_L0":>8s} {"R²_L1":>8s} {"Cum":>8s} {"R2_L1_delta":>12s}')
print("-" * 100)

for name, kw, al, R0, R1, cum in [
    ('SMACS 0723 (cluster)', sm_kw, sm_al, sm_R0, sm_R1, sm_cum),
    ('HUDF (deep field)',    hu_kw, hu_al, hu_R0, hu_R1, hu_cum),
    ('HST WFC3 HUDF',       ht_kw, ht_al, ht_R0, ht_R1, ht_cum),
]:
    cv_a = al.std()/al.mean() if al.mean() > 0 else 0
    delta = (1-R0)*R1  # additional variance explained by L1
    print(f'{name:<22s} {len(kw):>5d} {kw.mean():>7.0f}±{kw.std():>4.0f}  '
          f'{al.mean():>7.3f}±{al.std():>6.4f}  {cv_a:>7.4f}  '
          f'{R0:>7.3f}  {R1:>7.3f}  {cum:>7.3f}  {delta:>7.3f}')

print()
print("cv(α) = coefficient of variation of tail exponent = σ(α)/μ(α)")
print("R²_L1_delta = extra variance explained by α + spectral + position (above brightness alone)")
print()

# ═══ FIGURE ═══
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

datasets = [
    ('SMACS 0723\n(cluster)', sm_kw, sm_al, sm_R0, sm_R1, sm_cum, '#e41a1c'),
    ('HUDF JWST\n(deep field)', hu_kw, hu_al, hu_R0, hu_R1, hu_cum, '#377eb8'),
    ('HST WFC3\nHUDF', ht_kw, ht_al, ht_R0, ht_R1, ht_cum, '#4daf4a'),
]

# A: κ_W distributions
ax = axes[0, 0]
for name, kw, al, R0, R1, cum, c in datasets:
    ax.hist(kw, bins=20, alpha=0.5, label=f'{name.strip()}\nμ={kw.mean():.0f} σ={kw.std():.0f}', color=c)
ax.set_xlabel('κ_W'); ax.set_ylabel('count')
ax.set_title('A: κ_W distributions', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# B: Tail α distributions
ax = axes[0, 1]
for name, kw, al, R0, R1, cum, c in datasets:
    ax.hist(al, bins=20, alpha=0.5, label=f'{name.strip()}\nσ(α)={al.std():.4f}', color=c)
ax.set_xlabel('Tail α'); ax.set_ylabel('count')
ax.set_title('B: Tail α — field type discriminator', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# C: κ_W vs α scatter
ax = axes[0, 2]
for name, kw, al, R0, R1, cum, c in datasets:
    ax.scatter(kw, al, s=15, alpha=0.6, label=name.strip(), color=c, edgecolors='gray', linewidth=0.2)
ax.set_xlabel('κ_W'); ax.set_ylabel('Tail α')
ax.set_title('C: Phase space (κ_W, α)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# D: R² bar chart
ax = axes[1, 0]
x = np.arange(3); w = 0.25
for idx, (name, kw, al, R0, R1, cum, c) in enumerate(datasets):
    ax.bar(x[idx] - w, R0, w, color=c, alpha=0.7, label='L0: яркость' if idx==0 else '')
    ax.bar(x[idx], R1, w, color=c, alpha=0.35, hatch='//', label='L1: +α+спектр' if idx==0 else '')
    ax.bar(x[idx] + w, cum, w, color=c, alpha=0.55, label='Кумулятивный' if idx==0 else '')
    ax.text(x[idx], R0 + 0.03, f'{R0:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(x[idx] + w, cum + 0.03, f'{cum:.2f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax.set_xticks(x); ax.set_xticklabels([d[0] for d in datasets], fontsize=8)
ax.set_ylabel('R²'); ax.set_ylim(0, 1)
ax.set_title('D: Iterated regression R²', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')

# E: Residual σ compression
ax = axes[1, 1]
x2 = np.arange(3)
for idx, (name, kw, al, R0, R1, cum, c) in enumerate(datasets):
    if name.startswith('SMACS'):
        r0, r1, orig = sm_r0, sm_r1, sm_kw
    elif name.startswith('HUDF'):
        r0, r1, orig = hu_r0, hu_r1, hu_kw
    else:
        r0, r1, orig = ht_r0, ht_r1, ht_kw
    sigmas = [orig.std(), r0.std(), r1.std()]
    ax.plot([0, 1, 2], sigmas, 'o-', color=c, markersize=8, lw=2, label=name.strip())
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['κ_W raw', 'Resid 0\n(яркость)', 'Resid 1\n(+α+спектр)'], fontsize=8)
ax.set_ylabel('σ'); ax.legend(fontsize=8)
ax.set_title('E: σ compression through levels', fontsize=12, fontweight='bold')

# F: Summary table
ax = axes[1, 2]
ax.axis('off')
lines = [
    "FIELD TYPE vs SPECTRAL METRICS",
    "JWST only — identical setup",
    "─" * 34,
    "",
    "SMACS 0723 (galaxy cluster):",
    f"  n={len(sm_kw)}  κ_W: μ={sm_kw.mean():.0f} σ={sm_kw.std():.0f}",
    f"  α: μ={sm_al.mean():.3f} σ={sm_al.std():.3f}",
    f"  cv(α) = {sm_al.std()/sm_al.mean():.4f}",
    f"  R²_L0 = {sm_R0:.3f}",
    f"  R²_L1 = {sm_R1:.3f}  (+{(1-sm_R0)*sm_R1:.3f})",
    f"  Cum = {sm_cum:.3f}",
    "",
    "HUDF (deep field):",
    f"  n={len(hu_kw)}  κ_W: μ={hu_kw.mean():.0f} σ={hu_kw.std():.0f}",
    f"  α: μ={hu_al.mean():.3f} σ={hu_al.std():.3f}",
    f"  cv(α) = {hu_al.std()/hu_al.mean():.4f}",
    f"  R²_L0 = {hu_R0:.3f}",
    f"  R²_L1 = {hu_R1:.3f}  (+{(1-hu_R0)*hu_R1:.3f})",
    f"  Cum = {hu_cum:.3f}",
    "",
    "→ Tail α variance separates",
    "  cluster fields from deep fields.",
    "→ SMACS resists explanation (51%)",
    "  HUDF yields to it (85%).",
    "→ Lensing/clustering leaves",
    "  spectral fingerprint.",
]
ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=7.8,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#fffff5', alpha=0.95, edgecolor='gray', linewidth=0.5))

fig.suptitle('SMACS 0723 vs HUDF: two JWST fields, one spectral method\n'
             'κ_W + Tail α reveal field-type fingerprint',
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('smacs_vs_hudf.pdf', dpi=150, bbox_inches='tight')
fig.savefig('smacs_vs_hudf.png', dpi=150, bbox_inches='tight')
print('Saved smacs_vs_hudf.pdf + smacs_vs_hudf.png')
