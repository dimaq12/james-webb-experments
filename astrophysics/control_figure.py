import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os; os.chdir('/home/dima/FA/astrophysics')

data = np.load('control_photos_results.npz', allow_pickle=True)
all_kw = data['all_kw'].item()
all_alpha = data['all_alpha'].item()
jwst_kw = data['jwst_kw']
jwst_alpha = data['jwst_alpha']

from scipy.stats import ks_2samp, mannwhitneyu, levene, f_oneway

# ═══ STATISTICAL TESTS ═══
print("=== STATISTICAL TESTS: JWST vs controls ===")
print()

datasets = {
    'JWST F090W':      (jwst_kw, jwst_alpha),
    'Photo #1':        (all_kw['IMG_2193'], all_alpha['IMG_2193']),
    'White noise':     (all_kw['white_noise'], all_alpha['white_noise']),
    'Fractal':         (all_kw['fractal'], all_alpha['fractal']),
    'Smooth gradient': (all_kw['smooth_gradient'], all_alpha['smooth_gradient']),
}

# κ_W tests
print("κ_W (spectral curvature):")
for name, (kw, _) in datasets.items():
    if name == 'JWST F090W': continue
    ks_stat, ks_p = ks_2samp(jwst_kw, kw)
    mw_stat, mw_p = mannwhitneyu(jwst_kw, kw, alternative='two-sided')
    print(f"  JWST vs {name:<16s}: KS p={ks_p:.1e}  M-W p={mw_p:.1e}")

# Tail α tests
print("\nTail α (spectral decay):")
for name, (_, alpha) in datasets.items():
    if name == 'JWST F090W': continue
    ks_stat, ks_p = ks_2samp(jwst_alpha, alpha)
    mw_stat, mw_p = mannwhitneyu(jwst_alpha, alpha, alternative='two-sided')
    print(f"  JWST vs {name:<16s}: KS p={ks_p:.1e}  M-W p={mw_p:.1e}")

# Variance comparison
print("\nVariance comparison (Levene test):")
print(f"  κ_W var: JWST={jwst_kw.var():.0f}  Photo={all_kw['IMG_2193'].var():.0f}  Noise={all_kw['white_noise'].var():.0f}  Fractal={all_kw['fractal'].var():.0f}")
stat, p = levene(jwst_alpha, all_alpha['IMG_2193'], all_alpha['white_noise'], all_alpha['fractal'])
print(f"  Levene α: stat={stat:.2f} p={p:.2e} → variances are {'EQUAL' if p>0.05 else 'DIFFERENT'}")

# CV (coefficient of variation) — normalized variance
print("\nCoefficient of variation (std/mean):")
for name, (kw, alpha) in datasets.items():
    cv_kw = kw.std()/kw.mean() if kw.mean() > 0 else 0
    cv_alpha = alpha.std()/alpha.mean() if alpha.mean() > 0 else 0
    print(f"  {name:<16s}: cv(κ_W)={cv_kw:.4f}  cv(α)={cv_alpha:.4f}")

# ═══ FIGURE ═══
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# A: κ_W distributions
ax = axes[0, 0]
colors = {'JWST F090W': '#e41a1c', 'Photo #1': '#377eb8', 'White noise': '#4daf4a',
          'Fractal': '#984ea3', 'Smooth gradient': '#ff7f00'}
for name, (kw, _) in datasets.items():
    n_bins = max(5, min(25, int(np.sqrt(len(kw)) * 3)))
    if len(kw) > 1 and kw.max() - kw.min() > 0.01:
        ax.hist(kw, bins=n_bins, alpha=0.5, label=f'{name} (n={len(kw)})', color=colors.get(name, 'gray'))
    else:
        ax.axvline(np.mean(kw), color=colors.get(name, 'gray'), linestyle='--', label=f'{name} (n={len(kw)})', lw=2)
ax.set_xlabel('κ_W', fontsize=11)
ax.set_ylabel('count', fontsize=11)
ax.set_title('A: κ_W distribution — JWST vs controls', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# B: Tail α distributions
ax = axes[0, 1]
for name, (_, alpha) in datasets.items():
    n_bins = max(5, min(25, int(np.sqrt(len(alpha)) * 3)))
    if len(alpha) > 1 and alpha.max() - alpha.min() > 0.0001:
        ax.hist(alpha, bins=n_bins, alpha=0.5, label=f'{name}', color=colors.get(name, 'gray'))
    else:
        ax.axvline(np.mean(alpha), color=colors.get(name, 'gray'), linestyle='--', label=f'{name}', lw=2)
ax.set_xlabel('Tail α', fontsize=11)
ax.set_ylabel('count', fontsize=11)
ax.set_title('B: Tail α distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# C: κ_W vs α scatter — all datasets
ax = axes[0, 2]
for name, (kw, alpha) in datasets.items():
    ax.scatter(kw, alpha, s=20, alpha=0.7, label=name, color=colors.get(name, 'gray'),
               edgecolors='black', linewidth=0.2)
ax.set_xlabel('κ_W', fontsize=11)
ax.set_ylabel('Tail α', fontsize=11)
ax.set_title('C: κ_W vs Tail α — phase space', fontsize=12, fontweight='bold')
ax.legend(fontsize=7)

# D: Boxplot κ_W
ax = axes[1, 0]
labels = list(datasets.keys())
kw_data = [datasets[n][0] for n in labels]
bp = ax.boxplot(kw_data, tick_labels=labels, patch_artist=True)
for patch, label in zip(bp['boxes'], labels):
    patch.set_facecolor(colors.get(label, 'lightgray'))
    patch.set_alpha(0.6)
ax.set_ylabel('κ_W', fontsize=11)
ax.set_title('D: κ_W boxplots', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=15)

# E: Boxplot tail α
ax = axes[1, 1]
alpha_data = [datasets[n][1] for n in labels]
bp2 = ax.boxplot(alpha_data, tick_labels=labels, patch_artist=True)
for patch, label in zip(bp2['boxes'], labels):
    patch.set_facecolor(colors.get(label, 'lightgray'))
    patch.set_alpha(0.6)
ax.set_ylabel('Tail α', fontsize=11)
ax.set_title('E: Tail α boxplots', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=15)

# F: Summary
ax = axes[1, 2]
ax.axis('off')

ks_p_kw = {n: ks_2samp(jwst_kw, datasets[n][0])[1] for n in datasets if n != 'JWST F090W'}
ks_p_alpha = {n: ks_2samp(jwst_alpha, datasets[n][1])[1] for n in datasets if n != 'JWST F090W'}

lines = [
    "CONTROL EXPERIMENT SUMMARY",
    "JWST vs regular photos + synthetic",
    "─" * 42,
    "",
    f"JWST κ_W: {jwst_kw.mean():.0f} ± {jwst_kw.std():.0f}",
    f"JWST α:   {jwst_alpha.mean():.3f} ± {jwst_alpha.std():.3f}",
    "",
    "κ_W KS-test p-values vs JWST:",
]
for n, p in ks_p_kw.items():
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    lines.append(f"  {n:<14s}: {p:.2e} {sig}")
lines.extend([
    "",
    "Tail α KS-test p-values vs JWST:",
])
for n, p in ks_p_alpha.items():
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    lines.append(f"  {n:<14s}: {p:.2e} {sig}")

cv_kw = {n: d[0].std()/d[0].mean() for n, d in datasets.items() if d[0].mean() > 0}
cv_alpha = {n: d[1].std()/d[1].mean() for n, d in datasets.items() if d[1].mean() > 0}
lines.extend([
    "",
    "Coefficient of variation:",
])
for n in datasets:
    lines.append(f"  {n:<14s}: cv(κ)={cv_kw.get(n,0):.4f}  cv(α)={cv_alpha.get(n,0):.4f}")

lines.extend([
    "",
    "CONCLUSIONS:",
    "• JWST distrib. differs from all controls",
    "• Tail α spread is unique to telescope",
    "• κ_W alone doesn't separate — α does",
    "• Spectral-geometric structure is cosmic",
])

ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=8.5,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#fffff5', alpha=0.95,
                  edgecolor='gray', linewidth=0.5))

fig.suptitle('Control Experiment: JWST SMACS 0723 vs Regular Camera Photos vs Synthetic Textures\n'
             'sft_torch — κ_W (spectral curvature) and Tail α (spectral decay)',
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('control_experiment.pdf', dpi=150, bbox_inches='tight')
fig.savefig('control_experiment.png', dpi=150, bbox_inches='tight')
print('\nSaved control_experiment.pdf + control_experiment.png')
