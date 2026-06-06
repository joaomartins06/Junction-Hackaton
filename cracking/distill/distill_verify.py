import sys
import numpy as np
import torch
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

import quimb
import quimb.tensor as qtn
from qiskit import QuantumCircuit
from qiskit_quimb import quimb_circuit

qasm_path = sys.argv[1]
max_bond = int(sys.argv[2]) if len(sys.argv) > 2 else 128
n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

def to_backend(x):
    return torch.tensor(x, dtype=torch.complex64, device="cuda")

circuit = QuantumCircuit.from_qasm_file(qasm_path)
N = circuit.num_qubits
print(f"Circuit: {circuit.count_ops()}, N={N}")

# ============================================================
# RCM reordering: minimize MPS bandwidth
# ============================================================
adj = lil_matrix((N, N))
for instr in circuit.data:
    if len(instr.qubits) == 2:
        q0, q1 = instr.qubits[0]._index, instr.qubits[1]._index
        adj[q0, q1] = 1
        adj[q1, q0] = 1

perm = list(reverse_cuthill_mckee(adj.tocsr()))  # perm[new] = old
inv_perm = [0] * N
for new, old in enumerate(perm):
    inv_perm[old] = new

# Build reordered circuit
qc_re = QuantumCircuit(N)
for instr in circuit.data:
    new_q = [qc_re.qubits[inv_perm[q._index]] for q in instr.qubits]
    qc_re.append(instr.operation, new_q, instr.clbits)

print(f"RCM permutation applied. Reordered first 10: {perm[:10]}")

# ============================================================
# Run MPS on reordered circuit
# ============================================================
print(f"\n[1/2] MPS at bond={max_bond}...")
qc_mps = quimb_circuit(
    qc_re,
    quimb_circuit_class=qtn.CircuitMPS,
    to_backend=to_backend,
    max_bond=max_bond,
    cutoff=1e-12,
    progbar=True,
)

mapping = [qc_mps.qubits.index(q) for q in range(qc_mps.N)]
mapping = [mapping[q] for q in mapping]

print(f"\n[2/2] Sampling {n_samples}...")
samples_re = [''.join(bs[q] for q in mapping)
              for bs in qc_mps.sample(n_samples, seed=1234)]
arr_re = np.array([[int(s) for s in ss] for ss in samples_re])
bit_probs_re = arr_re.mean(axis=0)  # in reordered indexing

# Undo RCM: get probs in original qubit indexing
bit_probs = np.zeros(N)
for new, old in enumerate(perm):
    bit_probs[old] = bit_probs_re[new]

print(f"\nBit probs (original ordering): {bit_probs.round(3)}")

# Majority vote (Qiskit convention: rightmost = q0)
voted = (bit_probs > 0.5).astype(int)
bs_majority = ''.join(str(b) for b in voted[::-1])

# Confidence ranking
conf = np.abs(bit_probs - 0.5)
print(f"\nConfidence > 0.15: {(conf > 0.15).sum()}/{N}")
print(f"Confidence > 0.10: {(conf > 0.10).sum()}/{N}")

uncertain_sorted = sorted(range(N), key=lambda i: conf[i])
print(f"\nLeast confident qubits:")
for q in uncertain_sorted[:10]:
    print(f"  q{q}: p={bit_probs[q]:.3f}, conf={conf[q]:.3f}")

print(f"\n{'='*60}")
print(f"MAJORITY VOTE (RCM-MPS): {bs_majority}")
print(f"{'='*60}")