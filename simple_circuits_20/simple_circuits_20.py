import pennylane as qml
from pennylane import numpy as np
from numpy import pi, array
import sys



def simple_circuits_20(angle):
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
    prob = 0.0
    # QHACK #
    num_wires = 1
    dev = qml.device('default.qubit', wires=num_wires)

    # Step 2 : Create a quantum circuit and qnode
    @qml.qnode(dev)
    def rot_measure(theta):
        qml.RX(theta, wires = 0)
        return qml.probs(wires = [0])

    prob = array(rot_measure(angle))[0]

    # QHACK #
    return prob