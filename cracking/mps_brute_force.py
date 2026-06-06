from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# 1) Load QASM circuit
qasm_path = "/home/lucas/Downloads/challenge-40_35.qasm"
qc = QuantumCircuit.from_qasm_file(qasm_path)

# 2) Ensure we have measurements for shot-based sampling
qc.measure_all()

# 3) Configure MPS simulator
shots = 16384*8
bond_dim = 128  # try 32, 64, 128, ... (higher is usually more accurate)

sim = AerSimulator(
  method="matrix_product_state",
  matrix_product_state_max_bond_dimension=bond_dim,
)

# 4) Transpile + run
qc_t = transpile(qc, sim)
result = sim.run(qc_t, shots=shots).result()
counts = result.get_counts()

# 5) Most frequent bitstring as peak estimate
peak_bitstring = max(counts, key=counts.get)
peak_count = counts[peak_bitstring]
peak_prob_est = peak_count / shots

print("Estimated peak bitstring:", peak_bitstring)
print("Estimated peak probability:", peak_prob_est)

sorted_counts = sorted(
    counts.items(),
    key=lambda x: x[1],
    reverse=True
)

for bitstring, c in sorted_counts[:20]:
    print(bitstring, c, c/shots)