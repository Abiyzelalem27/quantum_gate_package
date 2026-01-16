
import numpy as np
from quantum_gate_package import (
    I, X, H, U_N_qubits, U_one_gate,
    U_two_gates, rho, P0, P1,evolve,
)

def test_projector_operators():
    assert np.allclose(P0 @ P0, P0)
    assert np.allclose(P1 @ P1, P1)
    
def test_I_gate():
    psi0 = np.array([1, 0], dtype=complex)
    psi1 = np.array([0, 1], dtype=complex)

    assert np.allclose(I @ psi0, psi0)
    assert np.allclose(I @ psi1, psi1)

def test_X_gate():
    psi0 = np.array([1, 0], dtype=complex)
    psi1 = np.array([0, 1], dtype=complex)

    assert np.allclose(X @ psi0, psi1)
    assert np.allclose(X @ psi1, psi0)

def test_H_gate():
    psi0 = np.array([1, 0], dtype=complex)

    result = H @ psi0
    expected = np.array([1, 1]) / np.sqrt(2)
    assert np.allclose(result, expected)


  

def test_CNOT_C_1_T_0():

    CNOT = np.kron(I, P0) + np.kron(X, P1)
    psi_11 = np.array([0, 0, 0, 1])
    result = CNOT @ psi_11

    expected = np.array([0, 1, 0, 0])
    assert np.allclose(result, expected)


def test_evolve_state_vector():

    psi = np.array([1, 0], dtype=complex)
    psi_out = evolve(psi, X)
    expected = np.array([0, 1], dtype=complex)
    assert np.allclose(psi_out, expected)


def test_evolve_density_matrix():
    
    psi = np.array([1, 0], dtype=complex)
    rho0 = np.outer(psi, psi.conj())

    rho_out = evolve(rho0, X)
    psi1 = np.array([0, 1], dtype=complex)
    expected = np.outer(psi1, psi1.conj())
    assert np.allclose(rho_out, expected)
    
    
def test_U_two_gates():
    N = 3
    i = 0
    j = 2
    U_two = U_two_gates(H, X, i, j, N)
    U_comp = U_one_gate(H, i, N) @ U_one_gate(X, j, N)
    assert np.allclose(U_two, U_comp)
