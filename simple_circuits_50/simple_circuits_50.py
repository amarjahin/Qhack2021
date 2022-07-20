
import pennylane as qml
from pennylane import numpy as np
from numpy import pi
import sys


def simple_circuits_50(angle):
    """The code you write for this challenge should be completely contained within this function
        between the # QHACK # comment markers.
    In this function:
        * Create the standard Bell State
        * Rotate the first qubit around the y-axis by angle
        * Measure the expectation value of the tensor observable `qml.PauliZ(0) @ qml.PauliZ(1)`
    Args:
        angle (float): how much to rotate a state around the y-axis
    Returns:
        float: the expectation value of the tensor observable
    """

    expectation_value = 0.0

    # QHACK #

    # Step 1 : initialize a device
    num_wires = 2
    dev = qml.device('default.qubit', wires=num_wires)

    # Step 2 : Create a quantum circuit and qnode
    @qml.qnode(dev)
    def rot_measure(theta):
        qml.Hadamard(wires = 0)
        qml.CNOT(wires=[0,1])
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(wires = [0])@qml.PauliZ(wires=[1]))

    # Step 3 : Run the qnode
    expectation_value = rot_measure(angle)
    print(rot_measure.draw())


    # QHACK #
    return expectation_value
