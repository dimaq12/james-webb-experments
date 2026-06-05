
"""
sparse_coding.py — Spectral compression via operator codec + recovery.
"""
import torch, sft_torch as sft, numpy as np
torch.set_printoptions(precision=4)
print("=" * 60)
print("SPECTRAL SPARSE CODING")
print("=" * 60)

N, M = 10, 5
A0 = torch.diag(torch.linspace(0.5, 3.0, N, dtype=torch.float64))
basis = torch.zeros((M, N, N), dtype=torch.float64)
for j in range(M):
    basis[j] = torch.diag(torch.sin(torch.linspace(0, (j+1)*np.pi/3, N, dtype=torch.float64)))

prog = sft.operator(A0, basis).compile("cpu")
fam = prog.family
codec = prog.codec()
print(f"N={fam.N} M={fam.M} rank={fam.W_rank}")
print(f"Codec capacity: {codec.capacity():.1f} bits")

rng = np.random.default_rng(42)
print("\n1. Compression quality on random k vectors:")
for i in range(5):
    k = torch.from_numpy(rng.normal(0, 1, M)).to(torch.float64)
    encoded = codec.encode(k)
    k_rec = codec.decode(encoded)
    err = float(torch.norm(k - k_rec))
    k_str = f"[{', '.join(f'{v:+.2f}' for v in k.tolist())}]"
    print(f"  {k_str:50s}  err={err:.2e}")

print("\n2. Compression ratio vs M:")
for M_test in [2, 4, 6, 8]:
    basis_t = torch.zeros((M_test, N, N), dtype=torch.float64)
    for j in range(M_test):
        basis_t[j] = torch.diag(torch.sin(torch.linspace(0, (j+1)*np.pi/3, N, dtype=torch.float64)))
    prog_t = sft.operator(A0, basis_t).compile("cpu")
    codec_t = prog_t.codec()
    raw = M_test * 8
    k_test = torch.from_numpy(rng.normal(0, 1, M_test)).to(torch.float64)
    encoded = codec_t.encode(k_test)
    k_rec = codec_t.decode(encoded)
    err = float(torch.norm(k_test - k_rec))
    print(f"  M={M_test}: raw={raw}B  encoded={len(encoded)}B  ratio={len(encoded)/raw*100:.0f}%  err={err:.2e}")

print("\n3. Batch prediction (16 k vectors -> spectra):")
DK = torch.from_numpy(rng.normal(0, 1, (16, M))).to(torch.float64)
preds = prog.predict_many(DK)
print(f"  shape: {tuple(preds.shape)}")

print("\n=== DONE ===")
