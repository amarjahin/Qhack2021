import pennylane as qml
from pennylane import numpy as np
from numpy import pi
import sys



def simple_circuits_30(angle):
    """The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.
    In this function:
        * Rotate the qubit around the x-axis by angle
        * Measure the probability the qubit is in the zero state
    Args:
        angle (float): how much to rotate a state around the x-axis
    Returns:
        float: the probability of of the state being in the 0 ground state
    """
    x_expectation = 0.0
    # QHACK #

    # Step 1 : initalize a device
    num_wires = 1
    dev = qml.device('default.qubit', wires=num_wires)

    # Step 2 : Create a quantum circuit and qnode
    @qml.qnode(dev)
    def rot_measure(theta):
        qml.RY(theta, wires = 0)
        return qml.expval(qml.PauliX(0))

    # Step 3 : Run the qnode
    

    x_expectation = rot_measure(angle)
    # print(rot_measure.draw())
    # QHACK #
    return x_expectation
