
"""
inverse_design.py — Target spectrum -> operator parameters (inverse problem).
Given a desired eigenvalue spectrum, find the k-parameters that produce it.
"""
import torch, sft_torch as sft, numpy as np
torch.set_printoptions(precision=4)
print("=" * 60)
print("INVERSE SPECTRAL DESIGN")
print("=" * 60)

N, M = 10, 5
A0 = torch.diag(torch.linspace(0.5, 4.0, N, dtype=torch.float64))
basis = torch.zeros((M, N, N), dtype=torch.float64)
for j in range(M):
    basis[j] = torch.diag(torch.sin(torch.linspace(0, (j+1)*np.pi/3, N, dtype=torch.float64)))

prog = sft.operator(A0, basis).compile("cpu")
fam = prog.family
print(f"N={fam.N} M={fam.M} rank={fam.W_rank} cond={fam.condition_number():.1f}")

k_true = torch.tensor([1.5, -2.0, 0.7, -0.5, 0.3], dtype=torch.float64)
target_lam = fam.spectrum(k_true)
print(f"\nk_true = {k_true.tolist()}")
print(f"target lambda[0..3] = {[f'{v:.3f}' for v in target_lam[:4].tolist()]}")

inv = prog.inverse(target_lam, steps=25, alpha=0.5)
k_found = inv.k
lam_achieved = prog.spectrum(k_found)
print(f"\nInverse: converged={inv.converged} error={inv.error:.2e}")
print(f"k_found = {[f'{v:.4f}' for v in k_found.tolist()]}")
print(f"k_error = {float(torch.norm(k_found - k_true)):.2e}")

print("\nSpectral shift tracking:")
for scale in [0.9, 1.0, 1.1, 1.3]:
    inv_s = prog.inverse(target_lam * scale, steps=15, alpha=0.5)
    print(f"  scale={scale:.1f}: steps={inv_s.steps} k=[{', '.join(f'{v:.2f}' for v in inv_s.k.tolist())}]")

codec = prog.codec()
y = codec.encode(k_found)
k_rec = codec.decode(y)
print(f"\nCodec round-trip error: {float(torch.norm(k_found - k_rec)):.2e}  capacity: {codec.capacity():.1f}bits")

print("\n=== DONE ===")
