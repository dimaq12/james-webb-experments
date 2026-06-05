
"""
llm_fingerprint.py — Text->transition operator->spectral signature.
Each token sequence becomes a transition matrix operator. Its spectral
properties (kw, alpha, omega) form a fixed-size fingerprint for the text.
"""
import torch, sft_torch as sft, numpy as np
from collections import Counter
torch.set_printoptions(precision=4)
print("=" * 60)
print("LLM SPECTRAL FINGERPRINTS")
print("=" * 60)

def token_to_operator(tokens, N=12):
    trans = torch.zeros((N, N), dtype=torch.float64)
    for i in range(len(tokens)-1):
        a, b = tokens[i] % N, tokens[i+1] % N
        trans[a, b] += 1.0
    trans += 0.1 * torch.eye(N, dtype=torch.float64)
    trans = trans / (trans.sum(dim=1, keepdim=True) + 1e-10)
    return trans

def fingerprint(tokens, N=12, M=4):
    A = token_to_operator(tokens, N)
    basis = [torch.eye(N, dtype=torch.float64)]
    for d in range(1, M):
        B = torch.zeros((N, N), dtype=torch.float64)
        for i in range(N):
            j = (i + d) % N; B[i,j] = 1.0; B[j,i] = 1.0
        basis.append(B)
    Bs = torch.stack(basis)
    prog = sft.operator(A, Bs).compile("cpu")
    fam = prog.family; tail = prog.tail(strict=False)
    kap = prog.kappa(method="hessian")
    return {
        "kw": float(kap.summary()["value"]),
        "alpha": float(tail.alpha if tail.alpha else 0),
        "omega": tail.omega.cpu().numpy(),
        "complexity": float(fam.complexity),
    }

# Fingerprint test sequences
seqs = {
    "Claude-like": [1,2,1,3,1,2,4,1,3,2,1,5,2,1,3],
    "GPT4-like":  [1,3,5,2,7,1,4,3,6,2,8,1,5,3,7],
    "repetitive": [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
}
rng = np.random.default_rng(99)
seqs["random"] = rng.integers(1, 8, 15).tolist()

print(f"\n{'Text':<14s} {'kw':>8s} {'alpha':>8s} {'|Omega|':>8s} {'complx':>8s}")
print("-" * 52)
for name, tokens in seqs.items():
    fp = fingerprint(tokens, N=12, M=4)
    omn = np.linalg.norm(fp["omega"])
    print(f"{name:<14s} {fp['kw']:>8.1f} {fp['alpha']:>8.4f} {omn:>8.4f} {fp['complexity']:>8.4f}")

print("\n2. Fingerprint robustness to perturbation")
base = [1,2,1,3,1,2,4,2,1]
fp_base = fingerprint(base, 12, 4)
kw_base = fp_base["kw"]
for label, tokens in [
    ("+1 repeat", base + [base[-1]]),
    ("+1 noise token", base + [99]),
    ("shuffled", list(rng.permutation(base))),
]:
    fp = fingerprint(tokens, 12, 4)
    shift = abs(fp["kw"] - kw_base)/(abs(kw_base)+1e-10)
    print(f"  {label:15s}: kw={fp['kw']:.0f} shift={shift:.1%}")

print("\n=== DONE ===")
