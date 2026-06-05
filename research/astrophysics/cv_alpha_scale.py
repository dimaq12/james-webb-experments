import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os; os.chdir('/home/dima/FA/astrophysics')

# All collected data points
data = [
    ('HUDF\n(ultra-deep)', 0.0026, 'JWST F150W', '#2166ac', 'deep'),
    ('SMACS blank\n(dark bg)', 0.0338, 'JWST F090W', '#92c5de', 'bg'),
    ('DSCF1809\n(outdoor photo)', 0.0779, 'camera', '#5e9ecf', 'photo'),
    ('IMG_2169\n(complex photo)', 0.0989, 'camera', '#4393c3', 'photo'),
    ('CEERS\n(wide survey)', 0.1430, 'JWST F277W', '#f4a582', 'galaxy'),
    ('SMACS F200W\n(cluster 2.0µm)', 0.1581, 'JWST F200W', '#d6604d', 'cluster'),
    ('SMACS F090W\n(cluster 0.9µm)', 0.1990, 'JWST F090W', '#b2182b', 'cluster'),
]

labels = [d[0] for d in data]
cv_vals = [d[1] for d in data]
filters = [d[2] for d in data]
colors = [d[3] for d in data]
categories = [d[4] for d in data]

fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [2, 1]})
plt.subplots_adjust(wspace=0.3)

# LEFT: bar chart — cv(α) ordered by morphological richness
ax = axes[0]
y_pos = range(len(data))
bars = ax.barh(y_pos, cv_vals, color=colors, edgecolor='gray', linewidth=0.5, height=0.6)

# value labels
for i, (v, f) in enumerate(zip(cv_vals, filters)):
    ax.text(v + 0.005, i, f'{v:.4f}  ({f})', va='center', fontsize=9)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('cv(α) = σ(α) / μ(α)', fontsize=12)
ax.set_title('Spectral Heterogeneity Scale\ncv(α) — coefficient of variation of tail exponent',
             fontsize=14, fontweight='bold')
ax.set_xlim(0, max(cv_vals) * 1.25)
ax.invert_yaxis()

# background bands
for i in range(0, len(data), 1):
    cat = categories[i]
    if cat == 'deep' or cat == 'bg':
        ax.axhspan(i-0.4, i+0.4, facecolor='#2166ac', alpha=0.06)
    elif cat == 'photo':
        ax.axhspan(i-0.4, i+0.4, facecolor='#4393c3', alpha=0.06)
    elif cat == 'galaxy':
        ax.axhspan(i-0.4, i+0.4, facecolor='#f4a582', alpha=0.08)
    elif cat == 'cluster':
        ax.axhspan(i-0.4, i+0.4, facecolor='#b2182b', alpha=0.08)

# category labels on right
cat_ranges = {}
for i, cat in enumerate(categories):
    if cat not in cat_ranges: cat_ranges[cat] = [i, i]
    else: cat_ranges[cat][1] = i

cat_labels = {
    'deep': 'empty sky',
    'bg': '',
    'photo': 'natural photos',
    'galaxy': 'galaxy surveys',
    'cluster': 'cluster fields',
}

# RIGHT: interpretation
ax = axes[1]
ax.axis('off')

lines = [
    "SPECTRAL HETEROGENEITY",
    "cv(α) = σ(α)/μ(α)",
    "─" * 30,
    "",
    "The tail exponent α measures",
    "how quickly eigenvalues decay.",
    "",
    "cv(α) captures how much α",
    "VARIES across a field.",
    "",
    "Hypothesis:",
    "cv(α) ∝ morphological richness",
    "      ∝ spectral heterogeneity",
    "",
    "─" * 30,
    "",
    "Observed ordering:",
    "",
    "cv≈0.00: ultra-deep empty field",
    "  (HUDF — same tail everywhere)",
    "",
    "cv≈0.03–0.10: natural photos",
    "  (sky+ground = moderate variety)",
    "",
    "cv≈0.14–0.20: rich fields",
    "  (CEERS: many galaxy types)",
    "  (SMACS: cluster + lensing)",
    "",
    "─" * 30,
    "",
    "NOT a cluster/deep binary.",
    "Looks like a continuous scale.",
    "",
    "Next: nebula → cv(α) = ?",
    "",
    "n=4 fields, 6 measurements",
    "identical operator setup throughout",
]
ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=9.5,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#fffff5', alpha=0.95,
                  edgecolor='gray', linewidth=0.5))

fig.suptitle('cv(α) — A Scale of Spectral Heterogeneity in Image Fields (sft_torch)',
             fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('cv_alpha_scale.png', dpi=150, bbox_inches='tight')
fig.savefig('cv_alpha_scale.pdf', dpi=150, bbox_inches='tight')
print('Saved cv_alpha_scale.png + cv_alpha_scale.pdf')
print(f'\nOrdered by cv(α):')
for d in sorted(data, key=lambda x: x[1]):
    print(f'  {d[1]:.4f}  {d[0].replace(chr(10)," ")}')
