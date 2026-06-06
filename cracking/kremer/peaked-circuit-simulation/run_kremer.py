from qiskit import QuantumCircuit
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit.transpiler import PassManager
from unswap import mpo_compress_unswap, mpo_to_mps
import numpy as np


def strip_meas(layer):
    new = QuantumCircuit(layer.num_qubits, layer.num_clbits)
    for instr in layer.data:
        if instr.operation.name not in ("measure", "barrier"):
            new.append(instr.operation, instr.qubits, instr.clbits)
    return new


def kremer_peak(qasm_path, max_bond=1024, cutoff=1e-8, unswap_threshold=1e6,
                center_ratio=0.5, top=5):
    """Run Kremer's MPO compression + unswapping on a QASM circuit.
    Returns: (peak_bitstring, peak_probability, top_n_list)
    Bit ordering: rightmost = qubit 0 (Qiskit convention).
    """
    circuit = QuantumCircuit.from_qasm_file(qasm_path)
    collect_2q = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)])
    circuit = collect_2q.run(circuit)
    N = circuit.num_qubits
    print(f"Circuit: {circuit.count_ops()}, N={N}")

    mpo_core, layers_left, layers_right, stats = mpo_compress_unswap(
        circuit,
        max_bond=max_bond,
        cutoff=cutoff,
        unswap_threshold=unswap_threshold,
        center_ratio=center_ratio,
    )

    layers_left = [strip_meas(l) for l in layers_left
                   if any(i.operation.name not in ("measure", "barrier") for i in l.data)]

    mps, final_perm = mpo_to_mps(
        mpo_core, layers_left, layers_right,
        max_bond=max_bond, cutoff=cutoff,
    )

    psi = np.asarray(mps.to_dense()).flatten()
    probs = np.abs(psi)**2
    probs = probs / probs.sum()

    # Use convention G: inverse permutation, reversed
    inv_perm = [final_perm.index(i) for i in range(N)]

    def decode(idx):
        site_bits = format(idx, f'0{N}b')
        qubit_bits = ['0'] * N
        for site_i, qubit_pos in enumerate(inv_perm):
            qubit_bits[qubit_pos] = site_bits[site_i]
        return ''.join(reversed(qubit_bits))

    ranked = sorted(enumerate(probs), key=lambda kv: -kv[1])
    top_list = [(decode(idx), p) for idx, p in ranked[:top]]

    print(f"\nTop {top} bitstrings (rightmost = q0):")
    for bs, p in top_list:
        print(f"  {bs}   p = {p:.6f}")

    peak_bs, peak_p = top_list[0]
    print(f"\nPEAK: {peak_bs}")
    print(f"Peak prob: {peak_p:.6f}")
    return peak_bs, peak_p, top_list


if __name__ == "__main__":
    import sys
    qasm = sys.argv[1]
    ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    bond = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    cutoff = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-6
    kremer_peak(qasm, max_bond=bond, cutoff=cutoff, center_ratio=ratio)