import os
from qiskit import transpile
from iqm.qiskit_iqm import IQMProvider

def get_iqm_backend(quantum_computer="garnet", mock=False):
    token = os.environ.get("IQM_TOKEN")
    if not token:
        raise ValueError("IQM_TOKEN not set")
    name = f"{quantum_computer}:mock" if mock else quantum_computer
    provider = IQMProvider("https://resonance.iqm.tech", token=token, quantum_computer=name)
    return provider.get_backend()

def transpile_for_iqm(circuit, backend):
    return transpile(circuit, backend=backend, layout_method='sabre', optimization_level=3)