import sys
import time
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
from qiskit import QuantumCircuit, transpile
from quimb.tensor import CircuitMPS
from qiskit_quimb import quimb_gates


def real(z):
    return float(z.real)


def build_backend(use_gpu, dtype_str, device):
    if not use_gpu:
        return None, "cpu (numpy)"
    import torch
    dt = torch.complex64 if dtype_str == "c64" else torch.complex128
    def to_backend(x):
        return torch.tensor(np.asarray(x), dtype=dt, device=device)
    info = f"{device}/{dtype_str} torch={torch.__version__} hip={torch.version.hip}"
    return to_backend, info


def simulate_mps(qc, max_bond, cutoff, to_backend, batch=200):
    gates = quimb_gates(qc)
    mps = CircuitMPS(N=qc.num_qubits, max_bond=max_bond, cutoff=cutoff,
                     to_backend=to_backend)
    t0 = time.time()
    for k in range(0, len(gates), batch):
        mps.apply_gates(gates[k:k+batch])
        done = min(k + batch, len(gates))
        print(f"  {done}/{len(gates)}  bond={mps.psi.max_bond()}  "
              f"t={time.time()-t0:.0f}s", flush=True)
    return mps.psi


def per_qubit_marginals(psi):
    n = psi.L
    probs = []
    for i in range(n):
        rho = psi.partial_trace_to_mpo(keep=[i]).to_dense()
        d = real(rho[0, 0]) + real(rho[1, 1])
        p0 = real(rho[0, 0]) / d if abs(d) > 1e-12 else 0.5
        probs.append(p0)
        print(f"  q{i:>3}  P(0)={p0:.4f}  bit={'0' if p0>=0.5 else '1'}  conf={abs(p0-0.5):.3f}", flush=True)
    return probs


def decode_qiskit(probs):
    n = len(probs)
    bits = ['0' if p >= 0.5 else '1' for p in probs]
    return ''.join(bits[i] for i in range(n - 1, -1, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qasm")
    ap.add_argument("--bond", type=int, required=True)
    ap.add_argument("--cutoff", type=float, default=1e-8)
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--opt", type=int, default=3)
    args = ap.parse_args()

    to_backend, dev_info = build_backend(not args.cpu, args.dtype, args.device)
    print(f"[bond {args.bond}] backend: {dev_info}", flush=True)

    qc = QuantumCircuit.from_qasm_file(args.qasm).remove_final_measurements(inplace=False)
    qc = transpile(qc, optimization_level=args.opt,
                   basis_gates=['u3', 'cx', 'rz', 'rx'])
    n = qc.num_qubits
    print(f"[bond {args.bond}] n={n}  gates={dict(qc.count_ops())}", flush=True)

    t0 = time.time()
    psi = simulate_mps(qc, args.bond, args.cutoff, to_backend)
    print(f"[bond {args.bond}] MPS done  t={time.time()-t0:.0f}s  "
          f"final_bond={psi.max_bond()}", flush=True)

    print(f"[bond {args.bond}] computing marginals...", flush=True)
    probs = per_qubit_marginals(psi)
    conf = [abs(p - 0.5) for p in probs]
    n_conf = sum(c > 0.3 for c in conf)

    bitstring = decode_qiskit(probs)
    print(f"[bond {args.bond}] mean|p-0.5|={np.mean(conf):.3f}  "
          f"min={min(conf):.3f}  confident(>0.3)={n_conf}/{n}", flush=True)
    print(f"[bond {args.bond}] PEAK: {bitstring}", flush=True)

    circuit_name = Path(args.qasm).stem
    out_path = f"peak_{circuit_name}_{args.bond}_{args.dtype}.txt"
    with open(out_path, "w") as fh:
        fh.write(bitstring + "\n")
    print(f"[bond {args.bond}] saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()