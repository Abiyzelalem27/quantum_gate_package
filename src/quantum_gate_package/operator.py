

import numpy as np

I = np.array([[1, 0],
              [0, 1]], dtype=complex)
X = np.array([[0, 1],
              [1, 0]], dtype=complex)
H = 1 / np.sqrt(2) * np.array([[1,  1],
                               [1, -1]], dtype=complex)

def U_one_gate(U, i, N):
    I_left = np.eye(2**i)
    I_right = np.eye(2**(N - i - 1))
    U_full = np.kron(np.kron(I_left, U), I_right)
    return U_full

def U_two_gates(U, V, i, j, N):
    I_left = np.eye(2**i)
    I_middle = np.eye(2**(j - i - 1))
    I_right = np.eye(2**(N - j - 1))
    Operator = np.kron(
        np.kron(
            np.kron(
                np.kron(I_left, U),
                I_middle
            ),
            V
        ),
        I_right
    )
    return Operator

def rho(states, probabilities):
    return sum(p * np.outer(psi, psi.conj())
        for psi, p in zip(states, probabilities))

def evolve_rho(U, rho):
    """
    Evolve a density matrix under a unitary operator.
    rho -> U rho U†
    """
    return U @ rho @ U.conj().T



