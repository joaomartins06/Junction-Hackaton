from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# 1) Load circuit from OpenQASM file
qasm_path = "circuits/challenge-8_27.qasm"
qc = QuantumCircuit.from_qasm_file(qasm_path)

# 2) Remove measurements (required for Statevector simulation)
qc_no_meas = qc.remove_final_measurements(inplace=False)

# 3) Compute statevector from the circuit acting on |0...0>
sv = Statevector.from_instruction(qc_no_meas)

# 4) Convert amplitudes to probabilities and get the peak bitstring
probs = sv.probabilities_dict()          # dict: bitstring -> probability
peak_bitstring = max(probs, key=probs.get)
peak_prob = probs[peak_bitstring]

print("Most likely bitstring:", peak_bitstring)
print("Peak probability:", peak_prob)