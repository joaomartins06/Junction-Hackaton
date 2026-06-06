import argparse
import sys
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
 
from qiskit import QuantumCircuit
 
 
def load_qasm(path):
    """Load OPENQASM 2.0 or 3.0. Qiskit 2.x prefers qasm2.load; fall back to from_qasm_file."""
    text = path.read_text()
    is_qasm3 = text.lstrip().startswith("OPENQASM 3")
    try:
        if is_qasm3:
            from qiskit import qasm3
            return qasm3.load(str(path))
        from qiskit import qasm2
        return qasm2.load(str(path))
    except Exception:
        return QuantumCircuit.from_qasm_file(str(path))
 
 
def summarize(qc):
    counts = qc.count_ops()
    n_2q = sum(v for k, v in counts.items() if k in ("cx", "cz", "swap", "rzz", "ecr", "iswap"))
    print("Qubits:        {}".format(qc.num_qubits))
    print("Depth:         {}".format(qc.depth()))
    print("Total gates:   {}".format(sum(counts.values())))
    print("2-qubit gates: {}".format(n_2q))
    print("Op breakdown:  {}".format(dict(counts)))
 
 
def draw(qc, out):
    """Draw circuit. Save to file if --out given, otherwise save as circuit.png."""
    import traceback
    try:
        fig = qc.draw(output="mpl", fold=80)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print("[draw] saved {}".format(out))
    except Exception:
        traceback.print_exc()
        print("[draw] falling back to text")
        print(qc.draw(output="text", fold=120))
 
 
def solve_statevector(qc, top):
    """Exact peak via statevector. Only feasible for ~<=28 qubits."""
    from qiskit.quantum_info import Statevector
 
    n = qc.num_qubits
    if n > 28:
        print("[statevector] N={} > 28 — refusing (would OOM). Use --mps instead.".format(n))
        return None
 
    qc_clean = qc.remove_final_measurements(inplace=False)
    if qc_clean is None:
        qc_clean = qc
 
    print("[statevector] simulating N={} ...".format(n))
    sv = Statevector.from_instruction(qc_clean)
    probs = sv.probabilities_dict()
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
 
    print("\nTop {} bitstrings (rightmost bit = qubit 0):".format(top))
    for bs, p in ranked[:top]:
        print("  {}   p = {:.6f}".format(bs, p))
 
    peak_bs, peak_p = ranked[0]
    print("\nPEAK bitstring: {}".format(peak_bs))
    print("Peak probability: {:.6f}".format(peak_p))
    return peak_bs
 
 
def solve_mps(qc, shots, bond, top):
    """Sample-based peak via Aer MPS simulator. For larger circuits."""
    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
    except ImportError:
        print("[mps] qiskit-aer not installed. pip install qiskit-aer", file=sys.stderr)
        sys.exit(1)
 
    qc = qc.copy()
    if not any(inst.operation.name == "measure" for inst in qc.data):
        qc.measure_all()
 
    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond,
    )
    print("[mps] shots={}, bond_dim={} ...".format(shots, bond))
    qc_t = transpile(qc, sim)
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
 
    print("\nTop {} bitstrings (rightmost bit = qubit 0):".format(top))
    for bs, c in ranked[:top]:
        print("  {}   p ~ {:.4f}   (count={})".format(bs, c / shots, c))
 
    peak_bs, peak_c = ranked[0]
    print("\nESTIMATED PEAK: {}".format(peak_bs))
    print("Estimated probability: {:.4f}".format(peak_c / shots))
    return peak_bs
 
 
def main():
    ap = argparse.ArgumentParser(description="Inspect / visualize / solve QASM peaked circuits.")
    ap.add_argument("qasm", type=Path, help="Path to .qasm file")
    ap.add_argument("--no-draw", action="store_true", help="Skip drawing the circuit")
    ap.add_argument("--no-solve", action="store_true", help="Skip peak-finding")
    ap.add_argument("--mps", action="store_true", help="Use MPS sampler (for N > ~28)")
    ap.add_argument("--shots", type=int, default=4096, help="Shots for MPS (default 4096)")
    ap.add_argument("--bond", type=int, default=64, help="MPS bond dimension (default 64)")
    ap.add_argument("--top", type=int, default=8, help="Show top-N bitstrings (default 8)")
    ap.add_argument("--out", type=Path, default=None, help="Output path for circuit diagram")
    args = ap.parse_args()
 
    if not args.qasm.exists():
        print("File not found: {}".format(args.qasm), file=sys.stderr)
        sys.exit(1)
 
    qc = load_qasm(args.qasm)
    print("=== {} ===".format(args.qasm))
    summarize(qc)
 
    if not args.no_draw:
        if args.out is None:
            png_dir = Path("png_circuits")
            png_dir.mkdir(exist_ok=True)
            args.out = png_dir / (args.qasm.stem + ".png")
        draw(qc, args.out)
 
    if args.no_solve:
        return
 
    print()
    if args.mps:
        solve_mps(qc, shots=args.shots, bond=args.bond, top=args.top)
    else:
        solve_statevector(qc, top=args.top)
 
 
if __name__ == "__main__":
    main()