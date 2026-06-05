
"""
omega_embeddings.py — Omega vectors as spectral feature embeddings.
For each 8x8 patch, compute Omega = (Omega(2), Omega(4), Omega(8), Omega(16))
and use it as a fixed-dimensional embedding for similarity comparison.
"""
import torch, sft_torch as sft, numpy as np
torch.set_printoptions(precision=4)

n, M = 8, 4; N = n * n
basis_tensors = []
for d in range(M):
    B = np.zeros((N, N))
    for i in range(N):
        xi, yi = i % n, i // n
        j = yi * n + ((xi + d) % n)
        B[i, j] = 1.0; B[j, i] = 1.0
    basis_tensors.append(torch.from_numpy(B))
Bs = torch.stack(basis_tensors).to(torch.float64)

def omega_embed(pixels):
    px = pixels.ravel() - pixels.min() + 0.1
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
    return tail.omega.cpu().numpy(), float(tail.alpha if tail.alpha else 0)

print("=" * 60)
print("OMEGA VECTOR EMBEDDINGS")
print("=" * 60)

rng = np.random.default_rng(42)
samples = {
    "smooth": np.linspace(0, 1, 64).reshape(8, 8),
    "noise":  rng.random(64).reshape(8, 8),
    "checker": np.tile([[0.2, 0.8], [0.8, 0.2]], (4, 4)),
    "gradient": np.tile(np.linspace(0, 1, 8), (8, 1)),
}

print("\n1. Omega embeddings for different textures")
embeddings = {}
for name, data in samples.items():
    om, al = omega_embed(data)
    embeddings[name] = om
    print(f"  {name:10s}  alpha={al:.3f}  Omega={[f'{v:+.4f}' for v in om]}")

print("\n2. Cosine similarity matrix")
names = sorted(samples.keys())
for n1 in names:
    for n2 in names:
        if n1 < n2:
            sim = np.dot(embeddings[n1], embeddings[n2]) / (np.linalg.norm(embeddings[n1])*np.linalg.norm(embeddings[n2])+1e-10)
            print(f"  cos({n1:10s}, {n2:10s}) = {sim:+.4f}")

print("\n3. Omega as structure detector: noise -> identity morph")
for frac in [0.0, 0.3, 0.7, 1.0]:
    data = rng.random(64).reshape(8, 8) * (1-frac) + np.eye(8)[:,:8].T[:8,:8] * frac
    om, al = omega_embed(data)
    print(f"  noise->id {frac:.0%}:  alpha={al:.3f}  norm(Omega)={np.linalg.norm(om):.4f}")

print("\n=== DONE ===")
