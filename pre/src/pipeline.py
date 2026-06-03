import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
import mlflow
import mlflow.sklearn

sys.path.insert(0, os.path.dirname(__file__))

from qubo import create_maxcut_graph, graph_to_qubo, classical_maxcut
from qaoa import run_qaoa
from mitigation import mitigated_expectation


def run_pipeline(n_nodes: int = 6, p: int = 1, shots: int = 1024,
                 use_mitigation: bool = False, seed: int = 42) -> dict:
    
    #put everything toegether 
    print(f"\n{'='*50}")
    print(f"PIPELINE: n={n_nodes}, p={p}, shots={shots}, mitigation={use_mitigation}")
    print(f"{'='*50}")

    G = create_maxcut_graph(n_nodes=n_nodes, seed=seed)
    print(f"\n[1] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    classical_best, classical_partition = classical_maxcut(G)
    print(f"    Classical Max-Cut: {classical_best}, partition: {classical_partition}")

    print(f"\n[2] Running QAOA (p={p})...")
    results = run_qaoa(G, p=p, shots=shots, seed=seed)
    approx_ratio = results['best_cut'] / classical_best
    print(f"    QAOA cut:            {results['best_cut']}")
    print(f"    Expectation:         {results['expectation']:.4f}")
    print(f"    Approximation ratio: {approx_ratio:.4f}")
    print(f"    Optimal params:      {np.round(results['optimal_params'], 3)}")

    if use_mitigation:
        print(f"\n[3] Applying ZNE mitigation...")
        gammas = results['optimal_params'][:p]
        betas = results['optimal_params'][p:]
        mitigated = mitigated_expectation(G, p=p, gammas=gammas, betas=betas, shots=shots)
        print(f"    Raw expectation:       {results['expectation']:.4f}")
        print(f"    Mitigated expectation: {mitigated:.4f}")
        results['mitigated_expectation'] = mitigated

    results['graph'] = G
    results['classical_best'] = classical_best
    results['approximation_ratio'] = approx_ratio
    results['n_nodes'] = n_nodes
    results['p'] = p

    return results


def plot_results(results: dict, save_path: str = None):
    #plot results
    G = results['graph']
    partition = results['best_partition']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    #graph visualized
    ax = axes[0]
    pos = nx.spring_layout(G, seed=42)
    colors = ['#2196F3' if p == 0 else '#F44336' for p in partition]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)

    cut_edges = [(i, j) for i, j in G.edges() if partition[i] != partition[j]]
    uncut_edges = [(i, j) for i, j in G.edges() if partition[i] == partition[j]]
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='green',
                           width=2.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, edge_color='gray',
                           style='dashed', ax=ax)
    ax.set_title(f"QAOA Max-Cut: {results['best_cut']}/{results['classical_best']} "
                 f"(ratio={results['approximation_ratio']:.3f})")
    ax.axis('off')

    #measurement histogram
    ax2 = axes[1]
    counts = results['counts']
    top10 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    bitstrings, freqs = zip(*top10)
    ax2.bar(range(len(bitstrings)), freqs, color='#2196F3')
    ax2.set_xticks(range(len(bitstrings)))
    ax2.set_xticklabels(bitstrings, rotation=45, fontsize=8)
    ax2.set_xlabel('Bitstring')
    ax2.set_ylabel('Counts')
    ax2.set_title('Top 10 Measurement Outcomes')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[4] Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    mlflow.set_experiment("qaoa-maxcut")

    with mlflow.start_run():
        params = dict(n_nodes=6, p=2, shots=2048, use_mitigation=True)
        mlflow.log_params(params)

        #run pipeline
        results = run_pipeline(**params)

        #Log metrics
        mlflow.log_metric("best_cut", results['best_cut'])
        mlflow.log_metric("classical_best", results['classical_best'])
        mlflow.log_metric("approximation_ratio", results['approximation_ratio'])
        mlflow.log_metric("expectation", results['expectation'])
        if 'mitigated_expectation' in results:
            mlflow.log_metric("mitigated_expectation", results['mitigated_expectation'])

        plot_path = "pre/results/maxcut_result.png"
        plot_results(results, save_path=plot_path)
        mlflow.log_artifact(plot_path)

        print(f"\n[MLflow] Run logged. View with: mlflow ui")