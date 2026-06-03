import numpy as np
import networkx as nx
from qiskit_aer import AerSimulator
from mitiq import zne
from qaoa import build_qaoa_circuit, compute_expectation


def scale_noise(circuit, scale_factor):
    #introduce noise
    from qiskit import QuantumCircuit
    if scale_factor == 1:
        return circuit
    new_qc = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for inst in circuit.data:
        new_qc.append(inst)
        if inst.operation.name == 'cx' and scale_factor >= 2:
            for _ in range(int(scale_factor) - 1):
                new_qc.append(inst)
                new_qc.append(inst)
    return new_qc


def mitigated_expectation(G: nx.Graph, p: int, gammas: list,
                           betas: list, shots: int = 1024,
                           scale_factors: list = [1, 2, 3]) -> float:

    simulator = AerSimulator(method='statevector')
    qc = build_qaoa_circuit(G, p, gammas, betas)

    expectations = []
    for scale in scale_factors:
        scaled_qc = scale_noise(qc, scale)
        job = simulator.run(scaled_qc, shots=shots)
        counts = job.result().get_counts()
        expectations.append(compute_expectation(counts, G))

    # Richardson extrapolation to zero noise
    coeffs = np.polyfit(scale_factors, expectations, deg=len(scale_factors)-1)
    mitigated = np.polyval(coeffs, 0)
    return float(mitigated)


if __name__ == "__main__":
    from qubo import create_maxcut_graph

    G = create_maxcut_graph(n_nodes=6)
    gammas = [0.4]
    betas = [0.3]

    simulator = AerSimulator()
    raw_counts = simulator.run(
        build_qaoa_circuit(G, 1, gammas, betas), shots=1024
    ).result().get_counts()
    raw = compute_expectation(raw_counts, G)
    mitigated = mitigated_expectation(G, p=1, gammas=gammas, betas=betas)

    print(f"Raw expectation:       {raw:.4f}")
    print(f"Mitigated expectation: {mitigated:.4f}")