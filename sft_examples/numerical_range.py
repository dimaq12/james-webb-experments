
"""
numerical_range.py — Numerical range, resolvent amplification, eigenbasis condition.
"""
import torch, sft_torch as sft, numpy as np
torch.set_printoptions(precision=3)
print("=" * 60)
print("NUMERICAL RANGE & PSEUDOSPECTRUM")
print("=" * 60)

for label, A in [
    ("Normal diag(1,2,3)", torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))),
    ("Skew-symmetric", torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 0.]], dtype=torch.float64)),
    ("Jordan-like", torch.tensor([[1., 0.3, 0.], [0., 1., 0.], [0., 0., 2.]], dtype=torch.float64)),
]:
    nd = sft.diagnostics.normality_defect(A)
    ec = sft.diagnostics.eigenbasis_condition(A)
    ev = torch.linalg.eigvals(A).real
    print(f"\n{label}:")
    print(f"  normality_defect = {nd:.4f}  eigenbasis_cond = {ec:.2f}")
    print(f"  eigenvalues = {[f'{v:.2f}' for v in sorted(ev.tolist())]}")

print("\n2. Numerical range proxy (field of values sample)")
for label, A in [
    ("Normal", torch.diag(torch.tensor([1., 3., 5.], dtype=torch.float64))),
    ("Perturbed", torch.tensor([[1., 0.5, 0.], [0., 3., 0.5], [0., 0., 5.]], dtype=torch.float64)),
]:
    nr = sft.diagnostics.numerical_range_proxy(A, samples=64)
    rad = nr.get('radius_estimate', float(list(nr.values())[0]) if nr else 0)
    print(f"  {label}: radius={rad:.3f}")

print("\n3. Resolvent amplification (pseudospectrum)")
A = torch.tensor([[1.0, 0.4, 0.0], [0.0, 2.0, 0.3], [0.0, 0.0, 3.0]], dtype=torch.float64)
ra = sft.diagnostics.resolvent_amplification(A, grid=20)
peak = ra.get('peak_resolvent', float(list(ra.values())[0]) if ra else 0)
print(f"  peak_resolvent = {peak:.2f}")

print("\n4. Normality vs condition growth")
for eps in [0.0, 0.1, 0.3, 0.5]:
    A = torch.diag(torch.linspace(1, 5, 5, dtype=torch.float64))
    for i in range(4): A[i, i+1] = eps
    nd = sft.diagnostics.normality_defect(A)
    ec = sft.diagnostics.eigenbasis_condition(A)
    print(f"  eps={eps:.1f}: nd={nd:.4f}  econd={ec:.1f}")

print("\n=== DONE ===")
