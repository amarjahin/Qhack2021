import sys
import pennylane as qml
from pennylane import numpy as np
from numpy import pi, zeros, array, matmul
from numpy.linalg import inv
from qiskit import QuantumCircuit, execute, Aer
# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)
# backend = Aer.get_backend('qasm_simulator')
backend = Aer.get_backend('statevector_simulator')


def non_var_qc(qc):
    qc.rx(a, qubit=0)
    qc.rx(b, qubit=1)
    qc.rx(a, qubit=1)
    qc.cx(0,1)
    qc.cx(1,2)
    qc.rz(a, qubit=0)
    qc.h(qubit=1)
    qc.cx(0,1)
    qc.rz(b, qubit=1)
    qc.h(qubit=0)

def non_var_inv_qc(qc):
    qc.h(qubit=0)
    qc.rz(-b, qubit=1)
    qc.cx(0,1)
    qc.h(qubit=1)
    qc.rz(-a, qubit=0)
    qc.cx(1,2)
    qc.cx(0,1)
    qc.rx(-a, qubit=1)
    qc.rx(-b, qubit=1)
    qc.rx(-a, qubit=0)

def var_qc(qc, params):
    non_var_qc(qc)
    qc.barrier()
    qc.rx(params[0], qubit=0)
    qc.ry(params[1], qubit=1)
    qc.rz(params[2], qubit=2)
    qc.barrier()
    non_var_qc(qc)
    qc.barrier()
    qc.rx(params[3], qubit=0)
    qc.ry(params[4], qubit=1)
    qc.rz(params[5], qubit=2)

def var_inv_qc(qc, params):
    qc.rz(-params[5], qubit=2)
    qc.ry(-params[4], qubit=1)
    qc.rx(-params[3], qubit=0)
    qc.barrier()
    non_var_inv_qc(qc)
    qc.barrier()
    qc.rz(-params[2], qubit=2)
    qc.ry(-params[1], qubit=1)
    qc.rx(-params[0], qubit=0)
    qc.barrier()
    non_var_inv_qc(qc)


def fsm_expval(params, i, j, t):
    s = zeros(len(params))
    s[i] = t[0]
    s[j] = t[1]
    qc = QuantumCircuit(3)
    var_inv_qc(qc, params)
    var_qc(qc, params + s*pi/2)
    result = execute(qc, backend=backend).result()
    expvals = abs(result.get_statevector()[0])**2
    return expvals

def fsmetric(params):
    fsm = zeros((len(params) ,len(params)))
    for i in range(len(params)):
        for j in range(len(params)):
            fsm[i, j] = (1/8)*(-fsm_expval(params, i, j, [1,1]) 
                + fsm_expval(params, i, j, [-1,1]) 
                + fsm_expval(params, i, j, [1,-1])  
                -fsm_expval(params, i, j, [-1,-1]) )
    return fsm

def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.
    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.
    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.
    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.
    Args:
        params (np.ndarray): Input parameters, of dimension 6
    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = zeros(6)
    fsm = fsmetric(array(params))
    fsm_inv = inv(fsm)
    dqnode = qml.grad(qnode, argnum=0)
    grad_f = dqnode(params)
    natural_grad = matmul(fsm_inv, grad_f)

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.
    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.
    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.
    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.
    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))

@qml.qnode(dev)
def qnode1(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.
    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.probs(wires=[0,1,2])


test_params_np = array([0,pi/2, pi/4, pi/8, 0, 0.1])
fsm = fsmetric(test_params_np)
fsm_inv = inv(fsm)

test_params = np.array([0,pi/2, pi/4, pi/8, 0, 0.1])
dqnode = qml.grad(qnode, argnum=0)
grad_f = dqnode(test_params)

qng = matmul(fsm_inv, grad_f)