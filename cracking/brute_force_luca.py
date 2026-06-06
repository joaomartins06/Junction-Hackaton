from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

qasm_path = "/home/joaomartins_06/Junction-Hackaton/cracking/circuits/challenge-48_37.qasm"
qc = QuantumCircuit.from_qasm_file(qasm_path)

qc.measure_all()

shots = 4096*32
bond_dim = 256

sim = AerSimulator(
    method="matrix_product_state",
    matrix_product_state_max_bond_dimension=bond_dim,
)

# IMPORTANT: no transpile(sim)
qc_t = qc

result = sim.run(qc_t, shots=shots).result()
counts = result.get_counts()

peak_bitstring = max(counts, key=counts.get)

print("Peak:", peak_bitstring)
print("Prob:", counts[peak_bitstring] / shots)

sorted_counts = sorted(
    counts.items(),
    key=lambda x: x[1],
    reverse=True
)

for bitstring, c in sorted_counts[:20]:
    print(bitstring, c, c/shots)