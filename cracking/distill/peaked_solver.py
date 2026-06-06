import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")
 
import numpy as np
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit_aer import AerSimulator
 
 
# ══════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════
 
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
 
def load_qasm(path):
    qc = qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    qc.remove_final_measurements(inplace=True)
    qc = qc.decompose(gates_to_decompose=["swap"])
    return qc
 
def submit_str(bits, n):
    """bits[i] = 0/1 for qubit i → challenge bitstring (rightmost = q[0])."""
    return "".join(str(int(bits[i])) for i in range(n - 1, -1, -1))
 
def marginals_from_counts(counts, n):
    """Per-qubit P(qubit=1) from Aer shot counts."""
    ones = [0] * n
    total = 0
    for bs, c in counts.items():
        bs = bs.replace(" ", "")
        total += c
        for i in range(n):
            if bs[n - 1 - i] == "1":
                ones[i] += c
    return [ones[i] / total for i in range(n)]
 
def bits_and_confidence(p1s):
    bits = [1 if p > 0.5 else 0 for p in p1s]
    conf = [abs(p - 0.5) for p in p1s]
    return bits, conf
 
def report(tag, n, bits, p1s, counts=None):
    conf = [abs(p - 0.5) for p in p1s]
    n_conf = sum(c > 0.30 for c in conf)
    weak = sorted(
        [(i, round(p1s[i], 4)) for i in range(n) if conf[i] <= 0.30],
        key=lambda x: abs(x[1] - 0.5),
    )
    submit = submit_str(bits, n)
    log(f"  [{tag}] confident={n_conf}/{n}  mean|p-.5|={np.mean(conf):.3f}  min={min(conf):.4f}")
    if counts:
        top = max(counts, key=counts.get)
        top_p = counts[top] / sum(counts.values())
        log(f"  [{tag}] top_freq={top_p:.4f}  top==majority? {top.replace(' ','') == submit}")
    log(f"  [{tag}] CANDIDATE: {submit}")
    if weak:
        log(f"  [{tag}] weak qubits: {weak[:10]}{'...' if len(weak) > 10 else ''}")
    return submit, n_conf, weak
 
 
# ══════════════════════════════════════════════════════════
#  Stage 1 — Aer MPS shot sampling
# ══════════════════════════════════════════════════════════
 
def aer_run(qc, bond, shots, seed=1):
    """Transpile + run Aer MPS; return counts and elapsed seconds."""
    qc_m = qc.copy(); qc_m.measure_all()
    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond,
        matrix_product_state_truncation_threshold=1e-12,
    )
    # bypass coupling-map limit for n >= 64
    qc_t = transpile(qc_m, basis_gates=["u3", "cx", "rz", "rx", "u1", "u2", "u"],
                     optimization_level=1)
    t0 = time.time()
    result = sim.run(qc_t, shots=shots, seed_simulator=seed).result()
    return result.get_counts(), time.time() - t0
 
 
def solve_aer(qc, bonds, shots, conf_thresh, conf_tol, seed=1):
    """
    Sweep bond dims, return (best_bits, best_p1s, converged).
    Stops early when bitstring is stable AND enough qubits are confident.
    """
    n = qc.num_qubits
    prev_submit = None
    best_bits, best_p1s = None, None
 
    for bond in bonds:
        log(f"  Aer bond={bond} shots={shots} ...")
        try:
            counts, dt = aer_run(qc, bond, shots, seed)
        except Exception as e:
            log(f"  Aer bond={bond} FAILED: {e}"); continue
 
        p1s = marginals_from_counts(counts, n)
        bits, conf = bits_and_confidence(p1s)
        n_conf = sum(c > conf_thresh for c in conf)
        submit, _, _ = report(f"aer b={bond} {dt:.0f}s", n, bits, p1s, counts)
 
        best_bits, best_p1s = bits, p1s
 
        converged = (n_conf >= n - conf_tol) and (submit == prev_submit)
        if converged:
            log(f"  Aer converged at bond={bond}")
            return best_bits, best_p1s, True
        prev_submit = submit
 
    return best_bits, best_p1s, False
 
 
# ══════════════════════════════════════════════════════════
#  Stage 2 — Quimb MPS marginals (direct P(0) via <Z>)
# ══════════════════════════════════════════════════════════
 
def quimb_run(qc, bond, cutoff=1e-8, batch=20):
    """Apply gates via quimb CircuitMPS; return per-qubit P(0) list."""
    from quimb.tensor import CircuitMPS
    from qiskit_quimb import quimb_gates
 
    qc_t = transpile(qc, optimization_level=3, basis_gates=["u3", "cx", "rz", "rx"])
    gates = quimb_gates(qc_t)
    n = qc_t.num_qubits
 
    mps = CircuitMPS(N=n, max_bond=bond, cutoff=cutoff)
    t0 = time.time()
    for k in range(0, len(gates), batch):
        mps.apply_gates(gates[k:k + batch])
        el = time.time() - t0
        if (k // batch) % 50 == 0:
            log(f"    quimb {min(k+batch,len(gates))}/{len(gates)} chi={mps.psi.max_bond()} t={el:.0f}s")
 
    psi = mps.psi
    Z = np.diag([1, -1]).astype(complex)
    terms = {(i,): Z for i in range(n)}
    zs = psi.compute_local_expectation_canonical(terms, return_all=True)
    p0s = [(1 + zs[(i,)].real) / 2 for i in range(n)]
    return p0s, psi, time.time() - t0
 
 
def quimb_hillclimb(psi, bits, sweeps=4):
    """Flip individual bits to locally maximise |<bits|psi>|^2."""
    def amp2(b):
        sel = {psi.site_ind(i): int(b[i]) for i in range(len(b))}
        return abs(psi.isel(sel).contract(all, optimize="auto-hq")) ** 2
 
    cur = list(bits); best = amp2(cur)
    log(f"  hillclimb start prob={best:.3e}")
    for s in range(sweeps):
        improved = False
        for i in range(len(cur)):
            t = cur.copy(); t[i] ^= 1
            p = amp2(t)
            if p > best:
                best, cur, improved = p, t, True
        log(f"    sweep {s+1}: prob={best:.3e}")
        if not improved: break
    return cur, best
 
 
def solve_quimb(qc, bonds, cutoff, conf_thresh, conf_tol, hillclimb=False):
    n = qc.num_qubits
    prev_submit = None
    best_bits, best_p1s = None, None
 
    for bond in bonds:
        log(f"  quimb bond={bond} ...")
        try:
            p0s, psi, dt = quimb_run(qc, bond, cutoff)
        except Exception as e:
            log(f"  quimb bond={bond} FAILED: {e}"); continue
 
        p1s = [1 - p for p in p0s]  # P(1) = 1 - P(0)
        bits, conf = bits_and_confidence(p1s)
        n_conf = sum(c > conf_thresh for c in conf)
        submit, _, _ = report(f"quimb b={bond} {dt:.0f}s", n, bits, p1s)
 
        best_bits, best_p1s = bits, p1s
 
        if hillclimb and n_conf >= n - 5:
            hc_bits, hc_prob = quimb_hillclimb(psi, bits)
            hc_submit = submit_str(hc_bits, n)
            log(f"  hillclimb -> {hc_submit}  prob={hc_prob:.3e}")
            if hc_submit != submit:
                changed = [i for i in range(n) if hc_bits[i] != bits[i]]
                log(f"  hillclimb CHANGED bits at {changed}")
            best_bits = hc_bits
            submit = hc_submit
 
        converged = (n_conf >= n - conf_tol) and (submit == prev_submit)
        if converged:
            log(f"  quimb converged at bond={bond}")
            return best_bits, best_p1s, True
        prev_submit = submit
 
    return best_bits, best_p1s, False
 
 
# ══════════════════════════════════════════════════════════
#  Stage 3 — Pauli propagation (independent Z expectation)
# ══════════════════════════════════════════════════════════
 
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0,1],[1,0]], complex)
_Y = np.array([[0,-1j],[1j,0]], complex)
_Z = np.diag([1,-1]).astype(complex)
_S = [_I2, _X, _Y, _Z]
 
def _pt1(U):
    return np.array([[0.5 * np.trace(_S[q] @ U.conj().T @ _S[p] @ U)
                      for q in range(4)] for p in range(4)], complex)
 
def _build_T_CX():
    CX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], complex)
    P2 = [np.kron(_S[a], _S[b]) for a in range(4) for b in range(4)]
    T = np.zeros((16,16), complex)
    for p in range(16):
        for q in range(16):
            T[p,q] = 0.25 * np.trace(P2[q] @ CX.conj().T @ P2[p] @ CX)
    return T
 
_T_CX = _build_T_CX()
 
def _gmat(name, params):
    if name in ("u3","u"):
        t,p,l = params
        return np.array([[np.cos(t/2), -np.exp(1j*l)*np.sin(t/2)],
                         [np.exp(1j*p)*np.sin(t/2), np.exp(1j*(p+l))*np.cos(t/2)]], complex)
    if name == "rz":
        t = params[0]
        return np.diag([np.exp(-1j*t/2), np.exp(1j*t/2)]).astype(complex)
    if name == "rx":
        t = params[0]; c,s = np.cos(t/2), np.sin(t/2)
        return np.array([[c,-1j*s],[-1j*s,c]], complex)
    if name == "x": return _X
    raise ValueError(f"unsupported gate: {name}")
 
def build_pauli_program(qc):
    """Reversed gate list as ('1q', q, T) or ('cx', ctrl, tgt)."""
    prog = []
    for inst in reversed(list(qc.data)):
        name = inst.operation.name
        if name in ("barrier", "measure"): continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if name == "cx":
            prog.append(("cx", qs[0], qs[1]))
        else:
            prog.append(("1q", qs[0], _pt1(_gmat(name, list(inst.operation.params)))))
    return prog
 
def pauli_expect_Z(prog, n, qubit, thresh=1e-6, cap=2_000_000):
    """Propagate Z_qubit backward; return <Z_qubit> and final term count."""
    op = {tuple(3 if k == qubit else 0 for k in range(n)): 1.0+0j}
    for g in prog:
        if g[0] == "cx":
            _, c, t = g; new = {}
            for pa, co in op.items():
                col = _T_CX[pa[c]*4 + pa[t]]
                for j in range(16):
                    if col[j] == 0: continue
                    qc_, qt_ = divmod(j, 4)
                    lst = list(pa); lst[c] = qc_; lst[t] = qt_
                    k = tuple(lst); new[k] = new.get(k, 0) + co * col[j]
            op = new
        else:
            _, q, T = g; new = {}
            for pa, co in op.items():
                p = pa[q]
                if p == 0:
                    new[pa] = new.get(pa, 0) + co; continue
                for qn in range(4):
                    if T[p, qn] == 0: continue
                    lst = list(pa); lst[q] = qn
                    k = tuple(lst); new[k] = new.get(k, 0) + co * T[p, qn]
            op = new
        if thresh > 0:
            op = {k: v for k,v in op.items() if abs(v) > thresh}
        if len(op) > cap:
            op = dict(sorted(op.items(), key=lambda kv: -abs(kv[1]))[:cap])
    val = sum(co for pa, co in op.items() if all(x in (0,3) for x in pa))
    return val.real, len(op)
 
def solve_pauli(qc, target_qubits=None, thresh=1e-6, cap=2_000_000, procs=1):
    """
    Compute <Z_i> for each qubit via Heisenberg propagation.
    Returns list of bits (0 if <Z_i>>=0 else 1).
    """
    from qiskit import transpile as _tr
    qc_t = _tr(qc, optimization_level=3, basis_gates=["u3","cx","rz","rx"])
    n = qc_t.num_qubits
    prog = build_pauli_program(qc_t)
 
    if target_qubits is None:
        target_qubits = list(range(n))
 
    results = {}
    t0 = time.time()
 
    if procs > 1:
        from multiprocessing import Pool
        def _init(p,nq,th,cp): 
            global _prog,_n,_thresh,_cap; _prog,_n,_thresh,_cap=p,nq,th,cp
        def _worker(q): 
            return q, *pauli_expect_Z(_prog, _n, q, _thresh, _cap)
        with Pool(procs, initializer=_init, initargs=(prog, n, thresh, cap)) as pool:
            for q, z, nt in pool.imap_unordered(_worker, target_qubits):
                bit = 0 if z >= 0 else 1
                results[q] = (z, bit, nt)
                log(f"  Pauli q{q:>3}: <Z>={z:+.4f} bit={bit} terms={nt} "
                    f"t={time.time()-t0:.0f}s")
    else:
        for q in target_qubits:
            z, nt = pauli_expect_Z(prog, n, q, thresh, cap)
            bit = 0 if z >= 0 else 1
            results[q] = (z, bit, nt)
            log(f"  Pauli q{q:>3}: <Z>={z:+.4f} bit={bit} terms={nt} "
                f"t={time.time()-t0:.0f}s")
 
    return results
 
 
# ══════════════════════════════════════════════════════════
#  Top-level per-circuit solver
# ══════════════════════════════════════════════════════════
 
def solve(qasm_path, bonds, shots, conf_thresh, conf_tol,
          do_quimb, hillclimb, do_pauli, pauli_only,
          pauli_thresh, pauli_cap, pauli_procs, out_dir):
 
    base = os.path.basename(qasm_path).replace(".qasm", "")
    log(f"{'='*60}")
    log(f"Circuit: {base}")
    log(f"{'='*60}")
 
    qc = load_qasm(qasm_path)
    n = qc.num_qubits
    log(f"n={n}  ops={dict(qc.count_ops())}")
 
    final_bits = None
    final_p1s = None
 
    # ── Aer MPS sweep ─────────────────────────────────────
    if not pauli_only:
        final_bits, final_p1s, converged = solve_aer(
            qc, bonds, shots, conf_thresh, conf_tol)
 
        if not converged:
            log("  Aer did not fully converge.")
 
        # ── Quimb fallback if still many weak qubits ──────
        if do_quimb and final_p1s is not None:
            n_weak = sum(abs(p-0.5) <= conf_thresh for p in final_p1s)
            if n_weak > conf_tol:
                log(f"  {n_weak} weak qubits -> trying quimb sweep")
                q_bits, q_p1s, q_conv = solve_quimb(
                    qc, bonds, cutoff=1e-8,
                    conf_thresh=conf_thresh, conf_tol=conf_tol,
                    hillclimb=hillclimb)
                if q_bits is not None:
                    # For weak qubits, prefer quimb result over aer if quimb is more confident
                    for i in range(n):
                        aer_conf = abs(final_p1s[i] - 0.5)
                        quimb_conf = abs(q_p1s[i] - 0.5)
                        if quimb_conf > aer_conf:
                            final_bits[i] = q_bits[i]
                            final_p1s[i] = q_p1s[i]
 
    # ── Pauli cross-check on remaining weak qubits ────────
    if do_pauli or pauli_only:
        if pauli_only or final_p1s is None:
            weak_qubits = list(range(n))
        else:
            weak_qubits = [i for i in range(n) if abs(final_p1s[i]-0.5) <= conf_thresh]
 
        if weak_qubits:
            log(f"  Pauli cross-check on {len(weak_qubits)} qubits ...")
            pauli_results = solve_pauli(qc, weak_qubits, pauli_thresh, pauli_cap, pauli_procs)
 
            if pauli_only:
                final_bits = [0] * n
            for q, (z, bit, nt) in pauli_results.items():
                if final_bits is not None and not pauli_only:
                    if final_bits[q] != bit:
                        log(f"  Pauli OVERRIDES q{q}: MPS bit={final_bits[q]} -> Pauli bit={bit} (<Z>={z:+.4f})")
                if final_bits is not None:
                    final_bits[q] = bit
            if final_p1s is None:
                final_p1s = [0.5] * n
                for q, (z, _, _) in pauli_results.items():
                    final_p1s[q] = (1 + z) / 2
 
    if final_bits is None:
        log(f"!!! {base}: no result produced"); return None
 
    # ── Final answer ──────────────────────────────────────
    submit = submit_str(final_bits, n)
    conf = [abs(p-0.5) for p in final_p1s]
    n_conf = sum(c > conf_thresh for c in conf)
    weak = [(i, round(final_p1s[i], 4)) for i in range(n) if conf[i] <= conf_thresh]
 
    log(f"  FINAL: {submit}")
    log(f"  confident={n_conf}/{n}  weak_qubits={weak}")
 
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_ANSWER.txt")
    with open(out_path, "w") as fh:
        fh.write(submit + "\n")
    log(f"  Wrote: {out_path}")
    return submit
 
 
# ══════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════
 
def main():
    ap = argparse.ArgumentParser(
        description="Peaked circuit peak-finder: Aer MPS + Quimb MPS + Pauli propagation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python peaked_circuit_solver.py circuit.qasm
  python peaked_circuit_solver.py *.qasm --bonds 64,128,256 --shots 4096
  python peaked_circuit_solver.py circuit.qasm --pauli               # Pauli on weak qubits only
  python peaked_circuit_solver.py circuit.qasm --pauli-only          # skip MPS entirely
  python peaked_circuit_solver.py circuit.qasm --quimb --hillclimb   # add quimb + hillclimb
""")
    ap.add_argument("qasm", nargs="+", help="QASM file(s)")
    ap.add_argument("--bonds",      default="64,128,256",
                    help="Bond dimension sweep, comma-separated (default: 64,128,256)")
    ap.add_argument("--shots",      type=int,   default=2048,
                    help="Aer shots per bond dim (default: 2048)")
    ap.add_argument("--conf-thresh",type=float, default=0.30,
                    help="|p-0.5| threshold for a qubit to be 'confident' (default: 0.30)")
    ap.add_argument("--conf-tol",   type=int,   default=2,
                    help="Max weak qubits allowed at convergence (default: 2)")
    ap.add_argument("--quimb",      action="store_true",
                    help="Fall back to quimb MPS for weak qubits after Aer sweep")
    ap.add_argument("--hillclimb",  action="store_true",
                    help="Run hillclimb refinement after quimb MPS (requires --quimb)")
    ap.add_argument("--pauli",      action="store_true",
                    help="Run Pauli propagation cross-check on weak qubits")
    ap.add_argument("--pauli-only", action="store_true",
                    help="Skip MPS entirely; use only Pauli propagation")
    ap.add_argument("--pauli-thresh", type=float, default=1e-6,
                    help="Truncation threshold for Pauli terms (default: 1e-6)")
    ap.add_argument("--pauli-cap",  type=int,   default=2_000_000,
                    help="Max Pauli terms kept per qubit (default: 2M)")
    ap.add_argument("--pauli-procs",type=int,   default=1,
                    help="Parallel processes for Pauli (default: 1; try 4-8 on server)")
    ap.add_argument("--out",        default="./results",
                    help="Output directory (default: ./results)")
    args = ap.parse_args()
 
    bonds = [int(x) for x in args.bonds.split(",")]
 
    all_results = {}
    for qasm_path in args.qasm:
        try:
            ans = solve(
                qasm_path, bonds, args.shots,
                args.conf_thresh, args.conf_tol,
                args.quimb, args.hillclimb,
                args.pauli, args.pauli_only,
                args.pauli_thresh, args.pauli_cap, args.pauli_procs,
                args.out,
            )
            all_results[qasm_path] = ans
        except KeyboardInterrupt:
            log("Interrupted."); break
        except Exception as e:
            log(f"!!! {qasm_path} FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results[qasm_path] = None
 
    log("═" * 60)
    log("SUMMARY")
    log("═" * 60)
    for path, ans in all_results.items():
        name = os.path.basename(path)
        log(f"  {name:35s} → {ans if ans else 'FAILED'}")
 
 
if __name__ == "__main__":
    main()