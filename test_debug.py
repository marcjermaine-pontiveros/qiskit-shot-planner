from qiskit.quantum_info import SparsePauliOp

op = SparsePauliOp("ZZ", coeffs=[1.0])
print(f"Attributes: {dir(op)}")
try:
    print(f"op.labels: {op.labels}")
except AttributeError as e:
    print(f"Error accessing labels: {e}")

try:
    print(f"op.paulis: {op.paulis}")
    print(f"op.paulis.to_labels(): {op.paulis.to_labels()}")
except Exception as e:
    print(f"Error accessing paulis: {e}")
