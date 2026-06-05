
"""
token_compare.py — Token sequence spectral distance for text comparison.
Converts token sequences into transition operators and compares them
via spectral metrics: kw-delta, alpha-delta, omega distance, cosine similarity.
"""
import torch, sft_torch as sft, numpy as np
torch.set_printoptions(precision=4)
print("=" * 60)
print("TOKEN SEQUENCE SPECTRAL DISTANCE")
print("=" * 60)

def token_to_operator(tokens, N=12):
    trans = torch.zeros((N, N), dtype=torch.float64)
    for i in range(len(tokens)-1):
        a, b = tokens[i] % N, tokens[i+1] % N
        trans[a, b] += 1.0
    trans += 0.1 * torch.eye(N, dtype=torch.float64)
    trans = trans / (trans.sum(dim=1, keepdim=True) + 1e-10)
    return trans

def finger(tokens, N=12, M=4):
    A = token_to_operator(tokens, N)
    basis = [torch.eye(N, dtype=torch.float64)]
    for d in range(1, M):
        B = torch.zeros((N, N), dtype=torch.float64)
        for i in range(N):
            j = (i+d) % N; B[i,j] = 1.0; B[j,i] = 1.0
        basis.append(B)
    prog = sft.operator(A, torch.stack(basis)).compile("cpu")
    tail = prog.tail(strict=False)
    kap = prog.kappa(method="hessian")
    kw = float(kap.summary()["value"])
    al = float(tail.alpha if tail.alpha else 0)
    om = tail.omega.cpu().numpy()
    return kw, al, om

def spectral_dist(t1, t2, N=12, M=4):
    kw1, al1, om1 = finger(t1, N, M)
    kw2, al2, om2 = finger(t2, N, M)
    return {
        "d_kw": abs(kw1 - kw2),
        "d_alpha": abs(al1 - al2),
        "d_omega": float(np.linalg.norm(om1 - om2)),
        "cos_omega": float(np.dot(om1, om2)/(np.linalg.norm(om1)*np.linalg.norm(om2)+1e-10)),
    }

seqs = {
    "A": [1,2,1,3,1,2,4,2,1,3,2,5,1,2,3],
    "B": [1,2,1,3,1,2,4,2,1,3,2,5,1,2,3],  # same as A
    "C": [1,2,1,3,1,2,4,2,1,3,2,7,1,2,3],  # 1 token changed
    "D": [5,6,7,5,8,7,6,5,9,8,7,6,5,8,7],  # different vocab
    "E": [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],  # repetitive
}

print("\n1. Omega distance matrix:")
for n1 in "ABCDE":
    row = [f"{spectral_dist(seqs[n1], seqs[n2])['d_omega']:.4f}" for n2 in "ABCDE"]
    print(f"  {n1}: {'  '.join(row)}")

print("\n2. Detailed pairwise comparison:")
for n1, n2 in [("A","B"), ("A","C"), ("A","D"), ("A","E")]:
    d = spectral_dist(seqs[n1], seqs[n2])
    print(f"  {n1} vs {n2}: d_kw={d['d_kw']:.1f} d_alpha={d['d_alpha']:.4f} d_omega={d['d_omega']:.4f} cos_omega={d['cos_omega']:+.4f}")

print("\n3. Adversarial perturbation test:")
base = [1,2,1,3,1,2,4,2,1,3]
rng = np.random.default_rng(7)
for label, tokens in [
    ("identical", base),
    ("last 3 shuffled", base[:7] + [base[9], base[8], base[7]]),
    ("random noise", rng.integers(1, 10, 10).tolist()),
    ("constant", [5]*10),
]:
    d = spectral_dist(base, tokens)
    print(f"  {label:20s}: d_omega={d['d_omega']:.4f}  cos_omega={d['cos_omega']:+.4f}")

print("\n=== DONE ===")
