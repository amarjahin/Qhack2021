import sys
import pennylane as qml
import numpy as np
from numpy import pi


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.
    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.
    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).
            * gradient is a real NumPy array of size (5,).
            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)
    # penny_grad = qml.grad(circuit, argnum=0)
    # print(penny_grad(weights))
    
    s = pi/4
    def grad_circ(w):
        grad = np.zeros([5], dtype=np.float64)
        for i in range(len(w)):
            s_vec = np.zeros(len(w))
            s_vec[i] = 1
            grad[i] = (circuit(w+s*s_vec) - circuit(w-s*s_vec))/(2*np.sin(s))
        return grad

    def hessian_circ(w):
        hes =  np.zeros([5, 5], dtype=np.float64)
        for i in range(len(w)):
            s_vec = np.zeros(len(w))
            s_vec[i] = 1
            hes[i, :] = (grad_circ(w+s*s_vec) - grad_circ(w-s*s_vec))/(2*np.sin(s))
        return hes 

    def hessian_circ_2(w):
        hes =  np.zeros([5, 5], dtype=np.float64)
        for i in range(len(w)):
            for j in range(len(w)):
                si = np.zeros(len(w))
                sj = np.zeros(len(w))
                si[i] = 1
                sj[j] = 1
                hes[i, j] = (circuit(w+s*(si + sj)) - circuit(w+s*(si-sj)) - circuit(w+s*(-si + sj)) + circuit(w+s*(-si-sj)))/(4*(np.sin(s)**2))
        return hes 
                

    gradient = grad_circ(weights)
    hessian = hessian_circ(weights)

    return gradient, hessian, circuit.diff_options["method"]

dev = qml.device("default.qubit", wires=3)
w_test = np.array([pi/4, 0, pi/4,pi/4,0])
a= gradient_200(weights=w_test, dev=dev)
