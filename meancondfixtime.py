# Script that calculates the mean conditional fixation time of a system  with respect to its p_+, 
# using both the Exact Linear Theory and a Gillespie Algorithm as shown by Ashcroft et al. (2014)

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

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
# 1. Exact Linear Theory (Matrix Solver)
# ==========================================
def solve_theory_matrix(p_plus, p_minus_val):
    num_states = (N - 1) * 2
    P = np.zeros((num_states, num_states))
    B_phi = np.zeros(num_states)

    def get_idx(i, env_type):
        return (i - 1) * 2 + env_type

    envs = [1, -1]
    ps = [p_plus, p_minus_val] 

    for i in range(1, N):
        for e_idx, sigma in enumerate(envs):
            row = get_idx(i, e_idx)
            p_sw = ps[e_idx]
            
            # 1. Calculate rates based on current sigma or flipped sigma
            w_p_stay, w_m_stay = get_rates(i, N, beta, sigma)
            w_p_sw, w_m_sw = get_rates(i, N, beta, envs[1 - e_idx])

            # 2. Compound probabilities if we stay in the same environment
            p_stay_up = (1 - p_sw) * w_p_stay
            p_stay_dn = (1 - p_sw) * w_m_stay
            p_stay_same = (1 - p_sw) * (1 - w_p_stay - w_m_stay)

            # 3. Compound probabilities if we switch environment
            p_sw_up = p_sw * w_p_sw
            p_sw_dn = p_sw * w_m_sw
            p_sw_same = p_sw * (1 - w_p_sw - w_m_sw)

            # Populate Discrete Transition Matrix P
            P[row, row] = p_stay_same
            P[row, get_idx(i, 1 - e_idx)] = p_sw_same

            if i > 1:
                P[row, get_idx(i - 1, e_idx)] = p_stay_dn
                P[row, get_idx(i - 1, 1 - e_idx)] = p_sw_dn
            
            if i < N - 1:
                P[row, get_idx(i + 1, e_idx)] = p_stay_up
                P[row, get_idx(i + 1, 1 - e_idx)] = p_sw_up
            else:
                # Transitions into absorbing state N
                B_phi[row] = p_stay_up + p_sw_up

    # Matrix solving logic
    I = np.eye(num_states)
    A = I - P
    
    # Solve for Fixation Probabilities (Phi)
    Phi_internal = np.linalg.solve(A, B_phi)
    
    # Solve for Conditional Expected Visits (Theta)
    Theta_internal = np.linalg.solve(A, Phi_internal)
    
    # Extract values for i=1
    # Note: index 0 is sigma=1 (plus), index 1 is sigma=-1 (minus)
    phi_1_plus = Phi_internal[get_idx(1, 0)]   
    phi_1_minus = Phi_internal[get_idx(1, 1)]  
    
    t_1_plus = Theta_internal[get_idx(1, 0)] / phi_1_plus if phi_1_plus > 1e-12 else np.nan
    t_1_minus = Theta_internal[get_idx(1, 1)] / phi_1_minus if phi_1_minus > 1e-12 else np.nan
    
    return t_1_plus, t_1_minus

# ==========================================
# 2. Simulation (Discrete Moran Process)
# ==========================================
def discrete_run(args):
    p_plus, p_minus_val, init_env = args
    import random
    i, env, steps = 1, init_env, 0
    
    while 0 < i < N:
        steps += 1
        # Swapped: Now checks if env is 1 to use p_plus (previously checked for -1)
        if random.random() < (p_plus if env == 1 else p_minus_val):
            env *= -1
            
        # Ensure get_rates is prepared to handle the swapped env values
        w_p, w_m = get_rates(i, N, beta, env)
        r = random.random()
        
        if r < w_p: i += 1
        elif r < w_p + w_m: i -= 1
            
    return (i == N, steps)

def simulate_system_time(p_plus, init_env, runs=3000):
    args = [(p_plus, p_minus, init_env) for _ in range(runs)]
    with mp.Pool() as pool:
        results = pool.map(discrete_run, args)
    
    fixation_times = [steps for fixated, steps in results if fixated]
    
    if len(fixation_times) == 0:
        return np.nan
        
    return np.mean(fixation_times)

# ==========================================
# 3. Execution and Plotting
# ==========================================
if __name__ == '__main__':
    p_plus_range = np.logspace(-4, 0, 40)
    theory_t_p, theory_t_m = [], []

    print("Solving Conditional Time Theory via Matrix Method...")
    for p in p_plus_range:
        tp, tm = solve_theory_matrix(p, p_minus)
        theory_t_p.append(tp)
        theory_t_m.append(tm)

    sim_p_plus = np.logspace(-4, 0, 12)
    sim_t_p, sim_t_m = [], []

    print("Running Simulations (Tracking Time)...")
    for p in sim_p_plus:
        print(f"  Simulating p+ = {p:.5f}")
        sim_t_p.append(simulate_system_time(p, 1, runs=5000))
        sim_t_m.append(simulate_system_time(p, -1, runs=5000))

    # Plotting
    plt.figure(figsize=(9, 6))
    plt.plot(p_plus_range, theory_t_p, 'b-', label='Theory (init: Coexistence)')
    plt.plot(p_plus_range, theory_t_m, 'r-', label='Theory (init: Coordination)')
    
    plt.scatter(sim_p_plus, sim_t_p, color='blue', marker='+', s=100, label='Sim (init: Coexistence)')
    plt.scatter(sim_p_plus, sim_t_m, color='red', marker='o', facecolors='none', s=60, label='Sim (init: Coordination)')

    plt.xscale('log')
    plt.xlabel(r'Switching probability $p_+$')
    plt.ylabel(r'Conditional fixation time $t_1$ (Steps)')
    plt.title('')
    plt.legend()
    plt.grid(False)
    plt.show()
