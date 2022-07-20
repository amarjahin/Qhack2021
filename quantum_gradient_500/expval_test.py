import sys
import pennylane as qml
import numpy as np
from numpy import pi


dev = qml.device("default.qubit", wires=3)
def observable():
    observable.name = 'op'
    qml.RX(pi/4, wires=0)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])

    qml.RY(pi/2, wires=1)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])

    qml.RX(pi/8, wires=2)

@qml.qnode(dev, interface=None)
def circuit():
    qml.PauliX(wires=1)
    qml.PauliX(wires=1)
    return qml.expval(observable)

print(circuit())