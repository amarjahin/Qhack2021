import sys
import pennylane as qml
from pennylane import numpy as np
from numpy import pi, zeros,zeros_like, sin


def parameter_shift(weights):
    """Compute the gradient of the variational circuit given by the
    ansatz function using the parameter-shift rule.
    Write your code below between the # QHACK # markers—create a device with
    the correct number of qubits, create a QNode that applies the above ansatz,
    and compute the gradient of the provided ansatz using the parameter-shift rule.
    Args:
        weights (array): An array of floating-point numbers with size (2, 3).
    Returns:
        array: The gradient of the variational circuit. The shape should match
        the input weights array.
    """

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(weights):
        for i in range(len(weights)):
            qml.RX(weights[i, 0], wires=0)
            qml.RY(weights[i, 1], wires=1)
            qml.RZ(weights[i, 2], wires=2)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 0])

        return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))

    penny_grad = qml.grad(circuit)
    print(penny_grad(weights))
    s = pi/2
    gradient = zeros_like(weights)
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            s_vec = np.array([[0, 0, 0],[0, 0, 0]])
            s_vec[i,j] = 1
            gradient[i,j] = (circuit(weights+s*s_vec) - circuit(weights-s*s_vec))/(2*sin(s))

    # QHACK #
    #
    # QHACK #

    return gradient

######################################
# remove this part before submission 
######################################

w_test = np.array([[pi/4, 0, pi/4], [pi/4, 0,0]])
a= parameter_shift(weights=w_test)

#######################################