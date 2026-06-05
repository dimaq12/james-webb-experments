
"""
exceptional_points.py — EP topology, winding numbers, eigenvalue tracking.
"""
import torch, sft_torch as sft, numpy as np
torch.set_printoptions(precision=3)
print("=" * 60)
print("EXCEPTIONAL POINTS & SPECTRAL TOPOLOGY")
print("=" * 60)

print("\n1. Eigenvalue approach to exceptional point")
for eps in [0.0, 0.3, 0.7, 1.0]:
    A = torch.tensor([[1.0, eps], [eps, 1.0]], dtype=torch.float64)
    ev = torch.linalg.eigvals(A).real
    gap = float(torch.abs(ev[0] - ev[1]))
    print(f"  eps={eps:.1f}  lambda=[{ev[0]:.3f}, {ev[1]:.3f}]  gap={gap:.3f}")

print("\n2. Eigenvalue tracking along parameter loop")
theta = torch.linspace(0, 2 * torch.pi, 13)
for t_val in [0.0, 1.57, 3.14, 4.71]:
    t = torch.tensor(t_val, dtype=torch.float64)
    A = torch.zeros((2, 2), dtype=torch.float64)
    A[0,0] = 1.0 + 0.3 * torch.cos(torch.tensor(t))
    A[1,1] = 1.0 + 0.3 * torch.cos(t + torch.pi/3)
    A[0,1] = 0.4 + 0.2 * torch.cos(t)
    A[1,0] = 0.4 + 0.2 * torch.sin(t)
    ev = torch.linalg.eigvals(A)
    gap = float(torch.abs(ev[0] - ev[1]))
    print(f"  theta={t:.2f}  gap={gap:.3f}")

print("\n3. Exceptional point detection on 3x3 family")
eps_vals = torch.linspace(0, 1.0, 11)
for eps in [0.0, 0.5, 1.0]:
    A = torch.diag(torch.linspace(1, 3, 3, dtype=torch.float64))
    A[0,1] = eps; A[1,2] = eps*0.7
    ev = torch.linalg.eigvals(A)
    gaps = [float(torch.abs(ev[i] - ev[j])) for i in range(3) for j in range(i+1,3)]
    min_gap = min(gaps)
    print(f"  eps={eps:.1f}  min_gap={min_gap:.4f}")

print("\n4. Winding number demo: rotate off-diagonal coupling")
for t_val in [0.0, 1.57, 3.14, 4.71, 6.28]:
    t = torch.tensor(t_val, dtype=torch.float64)
    A = torch.tensor([[1.0, 0.3*torch.cos(t)], [0.3*torch.sin(t), 2.0]], dtype=torch.float64)
    ev = torch.linalg.eigvals(A)
    print(f"  theta={t:.2f}  ev=[{ev[0].real:.3f}+{ev[0].imag:.3f}j, {ev[1].real:.3f}+{ev[1].imag:.3f}j]")

print("\n=== DONE ===")
