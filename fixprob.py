### Script for the third task of the SC project ###

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.sparse import linalg, eye, csr_matrix

# ==========================================
# Parameters
# ==========================================
N = 50           # Population size
beta = 0.5       # Intensity of selection 
p_minus = 0.01   # Fixed switching probability from +1 to -1

def get_payoffs(i, N, sigma):
    a, d = 1.0, 1.0
    b = 1.0 + 0.5 * sigma
    c = 1.0 + 0.9 * sigma
    pi_A = ((i - 1) * a + (N - i) * b) / (N - 1)
    pi_B = (i * c + (N - i - 1) * d) / (N - 1)
    return pi_A, pi_B

def get_rates(i, N, beta, sigma):
    pi_A, pi_B = get_payoffs(i, N, sigma)
    f_A, f_B = np.exp(beta * pi_A), np.exp(beta * pi_B)
    f_bar = (i * f_A + (N - i) * f_B) / N
    w_plus = (i * (N - i) / N**2) * (f_A / f_bar)
    w_minus = (i * (N - i) / N**2) * (f_B / f_bar)
    return w_plus, w_minus

# ==========================================
# 1. Exact Linear Theory 
# ==========================================
def solve_linear_theory_matrix(p_plus, p_minus_val):
    """
    Solves the coupled boundary value problem:
    w_i^+ phi_{i+1} + w_i^- phi_{i-1} + p (phi_i_opp - phi_i) - (w_i^+ + w_i^-) phi_i = 0
    using a global matrix solver.
    """
    # Number of internal states: (N-1) positions * 2 environments
    num_states = (N - 1) * 2
    A = np.zeros((num_states, num_states))
    B = np.zeros(num_states)

    def get_idx(i, env_type):
        # env_type: 0 for sigma=+1 (Coexist), 1 for sigma=-1 (Coord)
        return (i - 1) * 2 + env_type

    envs = [1, -1]
    ps = [p_minus_val, p_plus] # p_sigma: prob of leaving sigma

    for i in range(1, N):
        for e_idx, sigma in enumerate(envs):
            row = get_idx(i, e_idx)
            w_p, w_m = get_rates(i, N, beta, sigma)
            p_exit = ps[e_idx]

            # Diagonal: -(w+ + w- + p_exit)
            A[row, row] = -(w_p + w_m + p_exit)

            # Switching term: p_exit * phi_i in the OTHER environment
            A[row, get_idx(i, 1 - e_idx)] = p_exit

            # Step Back: w- * phi_{i-1}
            if i > 1:
                A[row, get_idx(i - 1, e_idx)] = w_m
            # (If i=1, phi_0 = 0, so no term added)

            # Step Forward: w+ * phi_{i+1}
            if i < N - 1:
                A[row, get_idx(i + 1, e_idx)] = w_p
            else:
                # If i = N-1, phi_N = 1. Move to the RHS (B vector)
                B[row] = -w_p

    # Solve the linear system A * Phi = B
    phi_internal = np.linalg.solve(A, B)
    
    # Return phi_1 for both environments
    return phi_internal[get_idx(1, 0)], phi_internal[get_idx(1, 1)]

# ==========================================
# 2. Simulation (Discrete Moran Process)
# ==========================================
def discrete_run(args):
    p_plus, p_minus_val, init_env = args
    import random
    i, env = 1, init_env
    while 0 < i < N:
        if random.random() < (p_plus if env == -1 else p_minus_val):
            env *= -1
        w_p, w_m = get_rates(i, N, beta, env)
        r = random.random()
        if r < w_p: i += 1
        elif r < w_p + w_m: i -= 1
    return (i == N)

def simulate_system(p_plus, init_env, runs=3000):
    args = [(p_plus, p_minus, init_env) for _ in range(runs)]
    with mp.Pool() as pool:
        results = pool.map(discrete_run, args)
    return sum(results) / runs

# ==========================================
# 3. Execution and Plotting
# ==========================================
if __name__ == '__main__':
    p_plus_range = np.logspace(-4, 0, 40)
    theory_p, theory_m = [], []

    print("Solving Theory via Matrix Method...")
    for p in p_plus_range:
        tp, tm = solve_linear_theory_matrix(p, p_minus)
        theory_p.append(tp)
        theory_m.append(tm)

    sim_p_plus = np.logspace(-4, 0, 12)
    sim_p, sim_m = [], []

    print("Running Simulations...")
    for p in sim_p_plus:
        print(f"  Simulating p+ = {p:.5f}")
        sim_p.append(simulate_system(p, 1, runs=4000))
        sim_m.append(simulate_system(p, -1, runs=4000))

    plt.figure(figsize=(8, 6))
    plt.plot(p_plus_range, theory_p, 'b-', label='Theory (init: Coexistence)')
    plt.plot(p_plus_range, theory_m, 'r-', label='Theory (init: Coordination)')
    plt.scatter(sim_p_plus, sim_p, color='blue', marker='+', s=100, label='Sim (init: Coexistence)')
    plt.scatter(sim_p_plus, sim_m, color='red', marker='o', facecolors='none', s=60, label='Sim (init: Coordination)')

    plt.xscale('log')
    plt.xlabel(r'Switching probability $p_+$')
    plt.ylabel(r'Fixation probability $\phi_1$')
    plt.title('')
    plt.legend()
    plt.grid(False)
    plt.show()
