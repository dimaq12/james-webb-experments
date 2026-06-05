
"""
spectral_algebra.py — Tensor sum, direct sum, Jordan fusion.
"""
import torch, sft_torch as sft, numpy as np
torch.set_printoptions(precision=3)
print("=" * 60)
print("SPECTRAL ALGEBRA")
print("=" * 60)

print("\n1. Tensor sum (Kronecker): H1(3x3) oplus_t H2(2x2) = 6x6")
H1 = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
H2 = torch.diag(torch.tensor([10.0, 20.0], dtype=torch.float64))
H_ts = torch.kron(H1, torch.eye(2, dtype=torch.float64)) + torch.kron(torch.eye(3, dtype=torch.float64), H2)
ev_ts = sorted(torch.linalg.eigvals(H_ts).real.tolist())
print(f"  H1+H2 ev: {[f'{v:.1f}' for v in ev_ts]}")
print(f"  Expected: lambda_i + mu_j = [11,12,21,22,13,23]")

print("\n2. Direct sum: block_diag")
H_ds = torch.block_diag(H1, H2)
ev_ds = sorted(torch.linalg.eigvals(H_ds).real.tolist())
print(f"  ev: {ev_ds}")

print("\n3. Jordan block J_4(2.0)")
J = sft.algebra.jordan_block(order=4).to(torch.float64)
print(J)
fp = sft.algebra.jordan_fingerprint(J)
print(f"  gm={fp.geometric_multiplicity}  am={fp.algebraic_multiplicity}")

print("\n4. Defect detection via Jordan fingerprint")
for eps in [0.0, 0.05, 0.3]:
    A = torch.diag(torch.ones(4, dtype=torch.float64))
    A[0, 1] = eps; A[1, 2] = eps * 0.5
    fp = sft.algebra.jordan_fingerprint(A)
    nd = sft.diagnostics.normality_defect(A)
    print(f"  eps={eps:.2f}: gm={fp.geometric_multiplicity}  nilp_index={fp.nilpotent_index}  nd={nd:.4f}")

print("\n=== DONE ===")
