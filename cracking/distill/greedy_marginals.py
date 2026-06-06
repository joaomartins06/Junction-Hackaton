import sys
import numpy as np
import torch
import quimb
import quimb.tensor as qtn
from qiskit import QuantumCircuit
from qiskit_quimb import quimb_circuit

qasm_path = sys.argv[1]
max_bond = int(sys.argv[2]) if len(sys.argv) > 2 else 256
n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 20000

def to_backend(x):
    return torch.tensor(x, dtype=torch.complex64, device="cuda")

circuit = QuantumCircuit.from_qasm_file(qasm_path)
N = circuit.num_qubits
print(f"Circuit: {circuit.count_ops()}, N={N}")

# ============================================================
# Run MPS, generate many samples
# ============================================================
print(f"\n[1/2] MPS at bond={max_bond}...")
qc_mps = quimb_circuit(
    circuit,
    quimb_circuit_class=qtn.CircuitPermMPS,
    to_backend=to_backend,
    max_bond=max_bond,
    cutoff=1e-12,
    progbar=True,
)
mapping = [qc_mps.qubits.index(q) for q in range(qc_mps.N)]
mapping = [mapping[q] for q in mapping]

print(f"\n[2/2] Sampling {n_samples} bitstrings...")
samples = [''.join(bs[q] for q in mapping)
           for bs in qc_mps.sample(n_samples, seed=1234)]
# Store as array indexed by qubit (q0 at column 0)
arr = np.array([[int(s[i]) for i in range(N)] for s in samples])
# arr[i, q] = bit at qubit q for sample i (q0 leftmost in the sample string... check below)

# ============================================================
# Greedy iterative bit fixing
# ============================================================
print(f"\nGreedy iterative marginal fixing:")
fixed = {}             # qubit_index -> 0 or 1
remaining = set(range(N))
mask = np.ones(len(arr), dtype=bool)  # which samples are still consistent

while remaining:
    filt = arr[mask]
    if len(filt) < 30:
        print(f"  Too few samples left ({len(filt)}), stopping greedy phase.")
        break

    marginals = filt.mean(axis=0)
    # Pick most biased unfixed bit
    best_q = max(remaining, key=lambda q: abs(marginals[q] - 0.5))
    p1 = marginals[best_q]
    bit = int(p1 > 0.5)
    conf = abs(p1 - 0.5)
    fixed[best_q] = bit
    remaining.remove(best_q)
    print(f"  Fix q{best_q}={bit} (p1={p1:.3f}, conf={conf:.3f}, samples={len(filt)})")

    # Update mask
    mask = mask & (arr[:, best_q] == bit)

# Fallback for unfixed: take marginal from full sample set
if remaining:
    print(f"\n{len(remaining)} bits unresolved by greedy. Falling back to global marginal:")
    global_marginals = arr.mean(axis=0)
    for q in remaining:
        bit = int(global_marginals[q] > 0.5)
        fixed[q] = bit
        print(f"  q{q}={bit} (global p1={global_marginals[q]:.3f})")

# Assemble bitstring (Qiskit convention: rightmost = q0)
bits = [fixed[q] for q in range(N)]
bs_greedy = ''.join(str(b) for b in bits[::-1])

# Compare to plain majority vote
plain_marginals = arr.mean(axis=0)
plain_voted = (plain_marginals > 0.5).astype(int)
bs_plain = ''.join(str(b) for b in plain_voted[::-1])

print(f"\n{'='*60}")
print(f"PLAIN MAJORITY:  {bs_plain}")
print(f"GREEDY:          {bs_greedy}")
diff = [i for i in range(N) if bits[i] != plain_voted[i]]
print(f"Differ at qubits: {diff}")
print(f"{'='*60}")