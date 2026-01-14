

import numpy as np
from quantum_gate_package.operator import (
    I,X,H,U_one_gate,
    U_two_gates,
    rho,
)

def test_identity_gate():
    psi0 = np.array([1, 0], dtype=complex)
    psi1 = np.array([0, 1], dtype=complex)

    assert np.allclose(I @ psi0, psi0)
    assert np.allclose(I @ psi1, psi1)


def test_x_gate():
    psi0 = np.array([1, 0], dtype=complex)
    psi1 = np.array([0, 1], dtype=complex)

    assert np.allclose(X @ psi0, psi1)
    assert np.allclose(X @ psi1, psi0)


def test_h_gate():
    psi0 = np.array([1, 0], dtype=complex)

    result = H @ psi0
    expected = np.array([1, 1]) / np.sqrt(2)

    assert np.allclose(result, expected)


def test_project_combined_gates():
    N = 3

    psi = np.zeros(2**N, dtype=complex)
    psi[0] = 1

    U = U_two_gates(H, X, i=0, j=1, N=N)
    final_state = U @ psi

    assert np.isclose(np.linalg.norm(final_state), 1.0)
