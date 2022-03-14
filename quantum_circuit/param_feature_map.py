from qiskit import (QuantumRegister, QuantumCircuit)
import matplotlib.pyplot as plt

def feature_map(nqubits, U, show=False):
    """
    Feature map circuit following Havlicek et al.
    
    nqubits  -> int, number of qubits, should match elements of input data
    U        -> gate returning QuantumCircuit object. Defines the feature map.
    show     -> visualise circuit
    """
    qbits = QuantumRegister(nqubits,'q')
    circuit = QuantumCircuit(qbits)

    #barriers just to make visualisation nicer
    circuit.h(qbits[:])
    circuit.barrier()
    circuit.append(U.to_instruction(),circuit.qubits)
    circuit.barrier()
    circuit.h(qbits[:])
    circuit.barrier()
    circuit.append(U.to_instruction(),circuit.qubits)
    circuit.barrier()

    if show:
        circuit.draw('mpl')
        plt.show()


    return circuit
