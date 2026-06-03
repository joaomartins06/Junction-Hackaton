import numpy as np
import networkx as nx
from typing import Optional


def create_maxcut_graph(n_nodes: int = 6, edge_prob: float = 0.5, seed: Optional[int] = 42) -> nx.Graph:
    #create a random graph
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    return G


def graph_to_qubo(G: nx.Graph) -> np.ndarray:
    #A QUBO formulations is definied by specifying the entries of Q_ij, for 
    #sum_i,j Q_ij x_i x_j, where x_i, x_jin {0,1} 
    #for the Max-Cut we maximize C = sum_{(i,j) in E} (x_i - x_j)^2 =
    # = sum_{(i,j) in E} (x_i + x_j - 2 x_i x_j) 
    #-> Q_ii = -1, Q_ij = 1 (if there is repetition of nodes)
    #Q could be triangular superior
    n = G.number_of_nodes()
    Q = np.zeros((n, n))

    for i, j in G.edges():
        Q[i][i] -= 1
        Q[j][j] -= 1
        Q[i][j] += 1
        Q[j][i] += 1

    return Q


def qubo_to_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    #QUBO (x in {0,1}) -> Ising (s in {-1,+1}), so x_i = (1 - s_i) / 2
    n = Q.shape[0]
    #off diagonal terms (interaction terms)
    J = np.zeros((n, n))
    #diagonal terms (local fields)
    h = np.zeros(n)
    offset = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            J[i][j] = (Q[i][j] + Q[j][i]) / 4
            J[j][i] = J[i][j]

        h[i] = -0.5 * (Q[i][i] + sum(Q[i][j] + Q[j][i] for j in range(n) if j != i) / 2)

    offset = 0.25 * np.sum(Q)

    return J, h, offset


def classical_maxcut(G: nx.Graph) -> tuple[int, list]:
    #classical brute-force solution for MAx-cut
    n = G.number_of_nodes()
    best_cut = 0
    best_partition = None

    for mask in range(1 << n):
        cut = 0
        for i, j in G.edges():
            if ((mask >> i) & 1) != ((mask >> j) & 1):
                cut += 1
        if cut > best_cut:
            best_cut = cut
            best_partition = [int((mask >> i) & 1) for i in range(n)]

    return best_cut, best_partition


if __name__ == "__main__":
    G = create_maxcut_graph(n_nodes=6)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    Q = graph_to_qubo(G)
    print(f"QUBO matrix shape: {Q.shape}")

    J, h, offset = qubo_to_ising(Q)
    print(f"Ising — J shape: {J.shape}, h shape: {h.shape}, offset: {offset:.3f}")

    best_cut, partition = classical_maxcut(G)
    print(f"Classical Max-Cut: {best_cut} cuts, partition: {partition}")