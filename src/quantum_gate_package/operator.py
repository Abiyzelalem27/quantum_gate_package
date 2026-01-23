

import numpy as np


# 1. BASIC SINGLE-QUBIT GATES

# Identity gate: leaves the qubit unchanged
I = np.array([[1, 0],
              [0, 1]])

# Pauli-X gate (quantum NOT)
# Swaps |0⟩ ↔ |1⟩
X = np.array([[0, 1],
              [1, 0]])

# Pauli-Y gate
# Rotation around the Y-axis
Y = np.array([[0, -1j],
              [1j,  0]])

# Pauli-Z gate
# Adds a phase of -1 to |1⟩
Z = np.array([[1,  0],
              [0, -1]])

# Hadamard gate (Creates superposition states)
H = 1 / np.sqrt(2) * np.array([[1,  1],
                               [1, -1]])

# Projector onto |0⟩
P0 = np.array([[1, 0],
               [0, 0]])

# Projector onto |1⟩
P1 = np.array([[0, 0],
               [0, 1]])

# Phase (S) gate
S = np.array([[1, 0],
              [0, 1j]])

# T gate
T = np.array([[1, 0],
              [0, np.e**(1j * np.pi / 4)]])


def projectors(dim):
    """
    Generate computational basis projectors {|i><i|} with the given dimension.
    """
    projectors = []
    for i in range(dim):
        ket = np.zeros(dim)
        ket[i] = 1
        P = np.outer(ket, ket)
        projectors.append(P)
    return projectors


# SINGLE-QUBIT ROTATION GATE

def rotation_gate(theta, n):
    """
    General single-qubit rotation gate.

    This function implements a unitary rotation of a single qubit
    by an angle `theta` around an axis `n` on the Bloch sphere.

    The rotation generator is constructed as N = n · σ,
    where σ = (X, Y, Z) are the Pauli matrices.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.
    n : tuple of floats
        Rotation axis (nx, ny, nz).
    """
    nx, ny, nz = n
    N = nx * X + ny * Y + nz * Z
    R = np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * N
    return R


# TWO-QUBIT GATE

# Controlled-NOT (CNOT) gate
# Control: first qubit
# Target: second qubit
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])


# MULTI-QUBIT OPERATOR CONSTRUCTION

def U_N_qubits(ops):
    """
    Constructs an N-qubit operator using tensor products.

    Parameters
    ----------
    ops : list of numpy.ndarray
        List of single-qubit operators.

    Returns
    -------
    numpy.ndarray
        N-qubit operator.
    """
    U = ops[0]
    for op in ops[1:]:
        U = np.kron(U, op)
    return U


def U_one_gate(V, i, N):
    """
    Applies a single-qubit gate to qubit i
    in an N-qubit system.

    Parameters
    ----------
    V : numpy.ndarray
        Single-qubit gate.
    i : int
        Target qubit index.
    N : int
        Total number of qubits.
    """
    ops = [I] * N
    ops[i] = V
    return U_N_qubits(ops)


def U_two_gates(V, W, i, j, N):
    """
    Applies two single-qubit gates to an N-qubit system.

    If i != j:
        applies V on qubit i and W on qubit j.

    If i == j:
        applies the composed gate V @ W on qubit i,
        preserving operator ordering.
    """
    ops = [I] * N

    if i == j:
        ops[i] = V @ W
    else:
        ops[i] = V
        ops[j] = W

    return U_N_qubits(ops)


# DENSITY MATRIX REPRESENTATION

def rho(states, probabilities):
    """
    Constructs a density matrix from pure states.

    Parameters
    ----------
    states : list of numpy.ndarray
        State vectors.
    probabilities : list of float
        Classical probabilities.
    """
    return sum(p * np.outer(psi, psi.conj())
               for psi, p in zip(states, probabilities))


# QUANTUM STATE EVOLUTION

def evolve(state, U):
    """
    Evolves a quantum state using a unitary operator.

    Parameters
    ----------
    state : numpy.ndarray
        State vector or density matrix.
    U : numpy.ndarray
        Unitary operator.
    """
    if state.ndim == 1:
        # Pure state evolution
        return U @ state
    elif state.ndim == 2:
        # Density matrix evolution
        return U @ state @ U.conj().T
    else:
        raise ValueError("State must be a vector or a density matrix")


def controlled_gate(U, control, target, N):
    """
    Controlled-U gate on an N-qubit register.

    Implements the projector decomposition:

        C_U = P0(control) ⊗ I  +  P1(control) ⊗ U(target)
    """
    if control == target:
        raise ValueError("Control and target must be different.")

    # Operator acting on the subspace where control qubit is |0⟩
    P0_ops = [
        P0 if i == control else I
        for i in range(N)
    ]

    # Operator acting on the subspace where control qubit is |1⟩
    P1_ops = [
        P1 if i == control else U if i == target else I
        for i in range(N)
    ]

    return U_N_qubits(P0_ops) + U_N_qubits(P1_ops)
