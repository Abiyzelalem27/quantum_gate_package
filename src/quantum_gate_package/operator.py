

import numpy as np

I = np.array([[1, 0],
              [0, 1]], dtype=complex)
X = np.array([[0, 1],
              [1, 0]], dtype=complex)
H = 1 / np.sqrt(2) * np.array([[1,  1],
             [1, -1]], dtype=complex)
P0 = np.array([[1, 0],
               [0, 0]], dtype=complex)
P1 = np.array([[0, 0],
               [0, 1]], dtype=complex)

def U_N_qubits(ops):
    U = ops[0]
    for op in ops[1:]:
        U = np.kron(U, op)
    return U 

def U_one_gate(V, i, N):
    ops = [I]*N
    ops[i] = V
    return U_N_qubits(ops)
    
def U_two_gates(V,W, i, j, N):
    ops = [I]*N
    ops[i] = V
    ops[j] = W
    return U_N_qubits(ops)

def rho(states, probabilities):
    return sum(p * np.outer(psi, psi.conj())
               for psi, p in zip(states, probabilities))

def evolve(state, U):
    if state.ndim == 1:
        return U @ state
    elif state.ndim == 2:
        return U @ state @ U.conj().T
    else:
        raise ValueError("State must be a vector or a density matrix")
        


