import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from typing import Optional
from qubo import create_maxcut_graph, graph_to_qubo, qubo_to_ising, classical_maxcut


def build_qaoa_circuit(G: nx.Graph, p: int, gammas: list, betas: list) -> QuantumCircuit:
    #QAOA circuit
    #there are a bunch of papers that present this paper
    #the mixer hamiltonian is just sum of X_i
    #the cost hamiltonian is the one from qaoa
    #not going to explain qaoa here

    n = G.number_of_nodes()
    qc = QuantumCircuit(n)

    #initial superposition H^n |0>^n
    qc.h(range(n))

    for layer in range(p):
        #exp(-i*gamma*H_C)
        for i, j in G.edges():
            qc.cx(i, j)
            qc.rz(2 * gammas[layer], j)
            qc.cx(i, j)

        #exp(-i*beta*H_M)
        for i in range(n):
            qc.rx(2 * betas[layer], i)

    qc.measure_all()
    return qc


def compute_expectation(counts: dict, G: nx.Graph) -> float:
    #compute the qaoa expectation value from the measurement counts
    total_shots = sum(counts.values())
    expectation = 0.0

    for bitstring, count in counts.items():
        x = [int(b) for b in reversed(bitstring)]
        cut = sum(1 for i, j in G.edges() if x[i] != x[j])
        expectation += cut * count

    return expectation / total_shots


def run_qaoa(G: nx.Graph, p: int = 1, shots: int = 1024,
             seed: Optional[int] = 42) -> dict:
    #straightforward 

    simulator = AerSimulator(method='statevector', seed_simulator=seed)
    n_params = 2 * p

    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        qc = build_qaoa_circuit(G, p, gammas, betas)
        job = simulator.run(qc, shots=shots)
        counts = job.result().get_counts()
        #put a minus as we will be minimizing
        return -compute_expectation(counts, G)

    np.random.seed(seed)
    x0 = np.random.uniform(0, np.pi, n_params)

    #optimize parameters 
    result = minimize(objective, x0, method='COBYLA',
                      options={'maxiter': 200, 'rhobeg': 0.5})

    #final run after optimization
    gammas_opt = result.x[:p]
    betas_opt = result.x[p:]
    qc_final = build_qaoa_circuit(G, p, gammas_opt, betas_opt)
    job = simulator.run(qc_final, shots=shots)
    counts = job.result().get_counts()

    best_bitstring = max(counts, key=counts.get)
    best_x = [int(b) for b in reversed(best_bitstring)]
    best_cut = sum(1 for i, j in G.edges() if best_x[i] != best_x[j])

    return {
        'optimal_params': result.x,
        'expectation': -result.fun,
        'best_cut': best_cut,
        'best_partition': best_x,
        'counts': counts,
        'converged': result.success,
    }


if __name__ == "__main__":
    G = create_maxcut_graph(n_nodes=6)
    classical_best, classical_partition = classical_maxcut(G)
    print(f"Classical Max-Cut: {classical_best}, partition: {classical_partition}")

    for p in [1, 2]:
        print(f"\nRunning QAOA with p={p}...")
        results = run_qaoa(G, p=p, shots=2048)
        print(f"  QAOA cut:      {results['best_cut']}")
        print(f"  Expectation:   {results['expectation']:.3f}")
        print(f"  Partition:     {results['best_partition']}")
        print(f"  Approximation ratio: {results['best_cut']/classical_best:.3f}")