import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

import quimb
import quimb.tensor as qtn
from qiskit import QuantumCircuit
from qiskit_quimb import quimb_circuit

qasm_path = sys.argv[1]
bonds = [32, 64, 128, 256, 512]
n_samples = 1000

def to_backend(x):
    return torch.tensor(x, dtype=torch.complex64, device="cuda")

circuit = QuantumCircuit.from_qasm_file(qasm_path)
N = circuit.num_qubits
name = qasm_path.split('/')[-1].replace('.qasm', '')
print(f"Circuit: {circuit.count_ops()}, N={N}")

# Run distillation at each bond, collect probs + samples
all_probs = {}
all_samples = {}

for bond in bonds:
    print(f"\n=== bond={bond} ===")
    qc_mps = quimb_circuit(
        circuit,
        quimb_circuit_class=qtn.CircuitPermMPS,
        to_backend=to_backend,
        max_bond=bond,
        cutoff=1e-12,
        progbar=True,
    )
    mapping = [qc_mps.qubits.index(q) for q in range(qc_mps.N)]
    mapping = [mapping[q] for q in mapping]
    samples = [''.join(bs[q] for q in mapping)
               for bs in qc_mps.sample(n_samples, seed=1234)]
    arr = np.array([[int(s) for s in ss] for ss in samples])
    all_probs[bond] = arr.mean(axis=0)
    all_samples[bond] = arr
    print(f"  marginals[:8] = {all_probs[bond][:8].round(3)}")
    del qc_mps
    torch.cuda.empty_cache()

# ============================================================
# Plot 1: bond-vs-qubit marginal heatmap
# ============================================================
prob_matrix = np.array([all_probs[b] for b in bonds])
fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(prob_matrix, aspect='auto', cmap='RdBu_r', vmin=0, vmax=1)
ax.set_yticks(range(len(bonds)))
ax.set_yticklabels(bonds)
ax.set_xlabel('Qubit index')
ax.set_ylabel('Bond dim')
ax.set_title(f'{name}: bit probability vs bond dim')
plt.colorbar(im, label='P(bit=1)')
plt.tight_layout()
plt.savefig(f'{name}_bond_heatmap.png', dpi=120)
print(f"\nSaved {name}_bond_heatmap.png")

# Convergence: stability of each bit across bonds
std_across_bonds = prob_matrix.std(axis=0)
mean_across_bonds = prob_matrix.mean(axis=0)
# A bit is "convergent + confident" if std is small AND mean is far from 0.5
convergent = std_across_bonds < 0.05
confident = np.abs(mean_across_bonds - 0.5) > 0.15
good = convergent & confident
print(f"\nConvergent bits: {convergent.sum()}/{N}")
print(f"Convergent AND confident bits: {good.sum()}/{N}")
print(f"Uncertain qubits (need attention): {sorted(np.where(~good)[0])}")

# Build candidate from convergent average
voted = (mean_across_bonds > 0.5).astype(int)
bs_convergent = ''.join(str(b) for b in voted[::-1])
print(f"\nCANDIDATE (bond-convergent vote): {bs_convergent}")

# ============================================================
# Plot 2: pairwise correlation matrix (using highest-bond samples)
# ============================================================
samples_arr = all_samples[bonds[-1]]
# Connected correlation: <b_i b_j> - <b_i><b_j>
mu = samples_arr.mean(axis=0)
C_conn = (samples_arr.T @ samples_arr) / len(samples_arr) - np.outer(mu, mu)

fig, ax = plt.subplots(figsize=(8, 7))
vmax = np.abs(C_conn).max()
im = ax.imshow(C_conn, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax.set_title(f'{name}: connected bit correlations (bond={bonds[-1]})')
ax.set_xlabel('Qubit j')
ax.set_ylabel('Qubit i')
plt.colorbar(im, label='<b_i b_j> - <b_i><b_j>')
plt.tight_layout()
plt.savefig(f'{name}_corr_heatmap.png', dpi=120)
print(f"Saved {name}_corr_heatmap.png")

# ============================================================
# Eigenvector-based candidate
# ============================================================
# Center samples around mean, find dominant correlation direction
centered = samples_arr - mu
cov = centered.T @ centered / len(centered)
eigvals, eigvecs = np.linalg.eigh(cov)
# Largest eigenvalue's eigenvector
v = eigvecs[:, -1]
print(f"\nTop eigenvalue: {eigvals[-1]:.4f}")
print(f"Eigenvector dominant components (top 8):")
top_components = np.argsort(-np.abs(v))[:8]
for q in top_components:
    print(f"  q{q}: v={v[q]:+.3f}, marginal={mu[q]:.3f}")

# Candidate from eigenvector: project mean shift
# bit = 1 if mu[i] + v[i] > 0.5 else 0 (interpretation: shift toward dominant mode)
# More robust: use sign of v to flip uncertain bits
voted_eig = voted.copy()
uncertain_idx = np.where(~good)[0]
for i in uncertain_idx:
    # If v[i] same sign as (mu[i] - 0.5), keep voted; else flip
    if (v[i] > 0) == (mu[i] > 0.5):
        voted_eig[i] = int(mu[i] > 0.5)
    else:
        voted_eig[i] = int(mu[i] < 0.5)
bs_eig = ''.join(str(b) for b in voted_eig[::-1])
print(f"\nCANDIDATE (eigenvector-adjusted): {bs_eig}")
print(f"  Differs from convergent vote at qubits: "
      f"{[i for i in range(N) if voted[i] != voted_eig[i]]}")

print("\n" + "="*60)
print("SUBMIT THESE (in order):")
print(f"  1. {bs_convergent}  (bond-convergent vote)")
print(f"  2. {bs_eig}         (eigenvector-adjusted)")
print("="*60)