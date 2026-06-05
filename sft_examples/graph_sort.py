
"""
graph_sort.py — Spectral graph algorithms via SFT: Laplacian, weight prediction, inverse.
"""
import torch, sft_torch as sft, numpy as np
print("=" * 60)
print("SPECTRAL GRAPH ALGORITHMS")
print("=" * 60)

n = 8
edges_path = torch.tensor([[i, i+1] for i in range(n-1)], dtype=torch.long).T
edges_star = torch.tensor([[0, i] for i in range(1, n)], dtype=torch.long).T
edges_cycle = torch.tensor([[i, (i+1)%n] for i in range(n)], dtype=torch.long).T

for name, edges in [("path", edges_path), ("star", edges_star), ("cycle", edges_cycle)]:
    prog = sft.graph(edges, n=n).laplacian().compile("cpu")
    fam = prog.family
    t = prog.tail(strict=False)
    kap = prog.kappa(method="hessian")
    print(f"\n{name.capitalize()} graph ({n} nodes):")
    print(f"  rank={fam.W_rank} complexity={fam.complexity:.3f} cond={fam.condition_number():.1f}")
    print(f"  kw={kap.summary()['value']:.1f}  alpha={t.alpha:.4f}")

# Graph comparison via Omega
graphs = {}
for name, edges in [("path", edges_path), ("star", edges_star), ("cycle", edges_cycle)]:
    prog = sft.graph(edges, n=n).laplacian().compile("cpu")
    t = prog.tail(strict=False)
    graphs[name] = t.omega

print("\nOmega distance matrix:")
for n1 in ["path", "star", "cycle"]:
    for n2 in ["path", "star", "cycle"]:
        if n1 < n2:
            d = float(torch.norm(graphs[n1] - graphs[n2]).cpu())
            print(f"  dist({n1}, {n2}) = {d:.4f}")

# Weight prediction
prog = sft.graph(edges_path, n=n).laplacian().compile("cpu")
w_uniform = torch.ones(prog.family.M, dtype=torch.float64)
w_skew = torch.ones(prog.family.M, dtype=torch.float64)
w_skew[0] = 2.0; w_skew[2] = 3.0
lam_u = prog.predict_weights(w_uniform).cpu()
lam_s = prog.predict_weights(w_skew).cpu()
print(f"\nWeight prediction (path):")
print(f"  uniform -> lam[:4]: {[f'{v:.2f}' for v in lam_u[:4].tolist()]}")
print(f"  skewed  -> lam[:4]: {[f'{v:.2f}' for v in lam_s[:4].tolist()]}")

# Inverse problem on graph
inv = prog.inverse(prog.family.lam0 * 1.2, steps=15, alpha=0.5)
print(f"\nInverse: converged={inv.converged} error={inv.error:.2e} steps={inv.steps}")

print("\n=== DONE ===")
