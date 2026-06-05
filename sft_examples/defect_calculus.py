"""
defect_calculus.py — Jordan fingerprint, non-normality, spectral phase transitions.
"""
import torch, sft_torch as sft, numpy as np
torch.set_printoptions(precision=4, sci_mode=False)

def normality_defect(A):
    return sft.diagnostics.normality_defect(A)

print("=" * 60)
print("SPECTRAL DEFECT CALCULUS")
print("=" * 60)

print("\n1. Jordan fingerprint scan (N=6, eps sweep)")
for eps in torch.linspace(0, 0.5, 11):
    A = torch.diag(torch.arange(1, 7, dtype=torch.float64))
    A[0, 1] = eps; A[1, 2] = eps * 0.5
    fp = sft.algebra.jordan_fingerprint(A)
    nd = normality_defect(A)
    print(f"  eps={eps:.2f}  gm={fp.geometric_multiplicity}  am={fp.algebraic_multiplicity}  nilp={fp.nilpotent_index}  nd={nd:.4f}")

print("\n2. Direct sum H1(3x3) + H2(2x2) = block_diag(5x5)")
H1 = torch.diag(torch.tensor([1., 2., 3.], dtype=torch.float64))
H2 = torch.diag(torch.tensor([4., 5.], dtype=torch.float64))
H_sum = torch.block_diag(H1, H2)
ev = sorted(torch.linalg.eigvals(H_sum).real.tolist())
print(f"  eigenvalues: {ev}")

print("\n3. Jordan block J_4(2.0)")
J = sft.algebra.jordan_block(order=4).to(torch.float64)
print(J)

print("\n4. Non-normality growth under off-diagonal perturbation")
for eps in [0.0, 0.1, 0.3, 0.5]:
    A = torch.diag(torch.ones(5, dtype=torch.float64))
    for i in range(4): A[i, i + 1] = eps
    nd = normality_defect(A)
    ec = sft.diagnostics.eigenbasis_condition(A)
    print(f"  eps={eps:.1f}  normality_defect={nd:.4f}  eigenbasis_cond={ec:.2f}")

print("\n5. Spectral phase transition: parametric family")
for t in torch.linspace(0, 2.0, 7):
    A = torch.zeros((4, 4), dtype=torch.float64)
    for i in range(4):
        A[i, i] = float(i) + t
        if i < 3: A[i, i + 1] = t * 0.3
    fp = sft.algebra.jordan_fingerprint(A)
    ev = sorted(torch.linalg.eigvals(A).real.tolist())
    print(f"  t={t:.1f}  ev={[f'{v:.1f}' for v in ev]}  gm={fp.geometric_multiplicity}  nd={normality_defect(A):.3f}")

print("\n=== DONE ===")
