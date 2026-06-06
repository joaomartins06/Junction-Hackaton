import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from collections import Counter
 
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit.transpiler import PassManager
 
from unswap import mpo_compress_unswap, mpo_to_mps
 

def strip_meas(layer):
    new = QuantumCircuit(layer.num_qubits, layer.num_clbits)
    for instr in layer.data:
        if instr.operation.name not in ("measure", "barrier"):
            new.append(instr.operation, instr.qubits, instr.clbits)
    return new
 
 
def load_and_prep(qasm_path):
    circuit = QuantumCircuit.from_qasm_file(qasm_path)
    collect_2q = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)])
    return collect_2q.run(circuit)
 
 
def run_kremer(circuit, center_ratio, max_bond, cutoff, unswap_threshold, top=10):
    """Single Kremer run at a given center_ratio. Returns dict with peak + top-N."""
    N = circuit.num_qubits
 
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
 
    inv_perm = [final_perm.index(i) for i in range(N)]
 
    def decode(idx):
        site_bits = format(idx, f'0{N}b')
        qubit_bits = ['0'] * N
        for site_i, qubit_pos in enumerate(inv_perm):
            qubit_bits[qubit_pos] = site_bits[site_i]
        return ''.join(reversed(qubit_bits))
 
    ranked = sorted(enumerate(probs), key=lambda kv: -kv[1])
    top_list = [(decode(idx), float(p)) for idx, p in ranked[:top]]
    return {
        "center_ratio": center_ratio,
        "peak": top_list[0][0],
        "peak_prob": top_list[0][1],
        "top": top_list,
        "n_qubits": N,
    }
 
 
def vote_per_qubit(results, N):
    """Take three Kremer results, vote bit-by-bit across them.
    Returns (consensus_bitstring, per_qubit_agreement_count).
    """
    peaks = [r["peak"] for r in results if r is not None]
    if not peaks:
        return None, None
    consensus = []
    agreement = []
    for i in range(N):
        # rightmost = q0, so bit at position N-1-i in the string corresponds to qi
        votes = Counter(p[N - 1 - i] for p in peaks)
        most_common_bit, count = votes.most_common(1)[0]
        consensus.append(most_common_bit)
        agreement.append(count)
    return ''.join(reversed(consensus)), agreement
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    ap = argparse.ArgumentParser(description="Run multi-split Kremer on a QASM circuit.")
    ap.add_argument("qasm", type=Path, help="Path to .qasm file")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.17, 0.5, 0.83],
                    help="Center ratios to try (default 0.17 0.5 0.83)")
    ap.add_argument("--bond", type=int, default=2048, help="Max bond dim (default 2048)")
    ap.add_argument("--cutoff", type=float, default=1e-6, help="SVD cutoff (default 1e-6)")
    ap.add_argument("--unswap-threshold", type=float, default=1e6,
                    help="MPO element threshold to trigger unswapping (default 1e6)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path (default: <qasm>_kremer.json)")
    args = ap.parse_args()
 
    if not args.qasm.exists():
        print(f"ERROR: file not found: {args.qasm}", file=sys.stderr)
        sys.exit(1)
 
    out_path = args.out or args.qasm.with_suffix(".kremer.json")
 
    print(f"=== Multi-Kremer on {args.qasm} ===")
    print(f"Ratios: {args.ratios}")
    print(f"max_bond={args.bond}, cutoff={args.cutoff}, unswap_threshold={args.unswap_threshold}")
    print(f"Output: {out_path}\n")
 
    circuit = load_and_prep(args.qasm)
    N = circuit.num_qubits
    print(f"Loaded: {dict(circuit.count_ops())}, N={N}\n")
 
    all_results = []
    summary = {
        "qasm": str(args.qasm),
        "n_qubits": N,
        "ratios": args.ratios,
        "bond": args.bond,
        "cutoff": args.cutoff,
        "results": [],
    }
 
    for ratio in args.ratios:
        print(f"\n{'='*60}")
        print(f"RUN: center_ratio = {ratio}")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        try:
            result = run_kremer(
                circuit, ratio,
                max_bond=args.bond,
                cutoff=args.cutoff,
                unswap_threshold=args.unswap_threshold,
            )
            result["elapsed_s"] = time.perf_counter() - t0
            all_results.append(result)
            summary["results"].append(result)
            print(f"\n  -> Peak: {result['peak']}  p={result['peak_prob']:.4f}  "
                  f"({result['elapsed_s']:.1f}s)")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception:
            print(f"\n!!! Split at {ratio} FAILED:")
            traceback.print_exc()
            all_results.append(None)
            summary["results"].append({"center_ratio": ratio, "error": traceback.format_exc()})
 
        # Save after EACH run so we don't lose progress
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  [checkpoint saved to {out_path}]")
 
    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    valid = [r for r in all_results if r is not None]
    print(f"\n\n{'='*60}")
    print(f"SUMMARY ({len(valid)}/{len(args.ratios)} runs succeeded)")
    print(f"{'='*60}\n")
 
    if not valid:
        print("All runs failed. Nothing to vote on.")
        sys.exit(1)
 
    print("Per-split peaks:")
    for r in valid:
        print(f"  ratio={r['center_ratio']:.2f}  ->  {r['peak']}  p={r['peak_prob']:.4f}")
 
    # Vote
    consensus, agreement = vote_per_qubit(valid, N)
    print(f"\nConsensus bitstring: {consensus}")
    print(f"Per-qubit agreement (qN-1 ... q0): {agreement}")
    print(f"  Bits where all {len(valid)} runs agree: "
          f"{sum(1 for a in agreement if a == len(valid))}/{N}")
 
    # If all runs agreed, that's the answer. Otherwise flag uncertain bits.
    uncertain = [i for i, a in enumerate(agreement) if a < len(valid)]
    if uncertain:
        # remember positions in agreement list are qN-1..q0 (right-to-left)
        uncertain_qubits = [N - 1 - i for i in uncertain]
        print(f"  Uncertain qubits (need verification): {sorted(uncertain_qubits)}")
        print(f"  -> {2**len(uncertain)} candidate bitstrings to enumerate if needed")
 
    summary["consensus"] = consensus
    summary["agreement"] = agreement
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFinal results: {out_path}")
    print(f"\nSUBMIT THIS BITSTRING: {consensus}")
 
 
if __name__ == "__main__":
    main()