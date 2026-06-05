import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os; os.chdir('/home/dima/FA/astrophysics')

data = [
    ('EMPTY SKY\n(synthetic constant)', 0.0017, 'synthetic', '#053061'),
    ('HUDF\n(ultra-deep JWST)', 0.0026, 'JWST', '#2166ac'),
    ('RIVER-LIKE\n(branching pattern)', 0.0073, 'synthetic', '#4393c3'),
    ('SMACS dark bg\n(blank field)', 0.0338, 'JWST', '#92c5de'),
    ('FOREST-LIKE\n(high-freq texture)', 0.0440, 'synthetic', '#5e9ecf'),
    ('CLOUDS\n(smooth gradient)', 0.0557, 'synthetic', '#6baed6'),
    ('CITY BLOCKS\n(grid pattern)', 0.0709, 'synthetic', '#9ecae1'),
    ('DSCF1809\n(outdoor photo)', 0.0779, 'camera', '#c6dbef'),
    ('MICROSCOPY\n(Voronoi cells)', 0.0807, 'synthetic', '#e0ecf4'),
    ('IMG_2169\n(complex photo)', 0.0989, 'camera', '#fddbc7'),
    ('CEERS\n(wide survey JWST)', 0.1430, 'JWST', '#f4a582'),
    ('SMACS F200W\n(cluster 2.0µm)', 0.1581, 'JWST', '#d6604d'),
    ('SOUTHERN RING\n(nebula JWST)', 0.1853, 'JWST', '#b2182b'),
    ('SMACS F090W\n(cluster 0.9µm)', 0.1990, 'JWST', '#67000d'),
]

labels = [d[0] for d in data]
cv_vals = [d[1] for d in data]
sources = [d[2] for d in data]
colors = [d[3] for d in data]

fig, ax = plt.subplots(1, 1, figsize=(10, 9))

y_pos = range(len(data))
bars = ax.barh(y_pos, cv_vals, color=colors, edgecolor='gray', linewidth=0.5, height=0.65)

for i, v in enumerate(cv_vals):
    src = sources[i]
    tag = f' [{src}]' if src != 'JWST' else ''
    ax.text(v + 0.004, i, f'{v:.4f}{tag}', va='center', fontsize=8,
            color='#333333')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8.5, family='monospace')
ax.set_xlabel('cv(α) = σ(α) / μ(α)', fontsize=13)
ax.set_xlim(0, max(cv_vals) * 1.35)
ax.invert_yaxis()

# color bands for categories
bands = [
    (0.000, 0.010, 'empty / uniform', '#053061'),
    (0.030, 0.100, 'natural scenes', '#6baed6'),
    (0.140, 0.200, 'rich structure\n(clusters, nebulae, surveys)', '#b2182b'),
]
for lo, hi, label, bcolor in bands:
    ax.axvspan(lo, hi, facecolor=bcolor, alpha=0.06, zorder=0)
    ax.text(lo + (hi-lo)/2, -1.2, label, ha='center', fontsize=8,
            fontstyle='italic', color=bcolor, fontweight='bold')

ax.set_title('cv(α) — Spectral Heterogeneity Index\n14 measurements across real & synthetic fields',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig('cv_alpha_full_scale.png', dpi=150, bbox_inches='tight')
fig.savefig('cv_alpha_full_scale.pdf', dpi=150, bbox_inches='tight')
print('Saved cv_alpha_full_scale.png + .pdf')
