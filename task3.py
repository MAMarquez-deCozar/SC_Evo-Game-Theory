### Script combinado para probabilidad de fijación y tiempo condicional medio ###

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import random

# ==========================================
# Parámetros Globales
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
# 1. Métodos Analíticos (Matrices)
# ==========================================

def solve_prob_matrix(p_plus, p_minus_val):
    """ Matrix solver for fixation probability (Continuous BVP) """
    num_states = (N - 1) * 2
    A = np.zeros((num_states, num_states))
    B = np.zeros(num_states)

    def get_idx(i, env_type):
        return (i - 1) * 2 + env_type

    envs = [1, -1]
    ps = [p_minus_val, p_plus]

    for i in range(1, N):
        for e_idx, sigma in enumerate(envs):
            row = get_idx(i, e_idx)
            w_p, w_m = get_rates(i, N, beta, sigma)
            p_exit = ps[e_idx]

            A[row, row] = -(w_p + w_m + p_exit)
            A[row, get_idx(i, 1 - e_idx)] = p_exit

            if i > 1:
                A[row, get_idx(i - 1, e_idx)] = w_m
            if i < N - 1:
                A[row, get_idx(i + 1, e_idx)] = w_p
            else:
                B[row] = -w_p

    phi_internal = np.linalg.solve(A, B)
    return phi_internal[get_idx(1, 0)], phi_internal[get_idx(1, 1)]


def solve_time_matrix(p_plus, p_minus_val):
    """ Matrix solver for conditional fixation time (Discrete Transition) """
    num_states = (N - 1) * 2
    P = np.zeros((num_states, num_states))
    B_phi = np.zeros(num_states)

    def get_idx(i, env_type):
        return (i - 1) * 2 + env_type

    envs = [1, -1]
    # CORREGIDO: Tasa de salida de entorno 1 es p_minus, tasa de salida de -1 es p_plus
    ps = [p_minus_val, p_plus] 

    for i in range(1, N):
        for e_idx, sigma in enumerate(envs):
            row = get_idx(i, e_idx)
            p_sw = ps[e_idx]
            
            w_p_stay, w_m_stay = get_rates(i, N, beta, sigma)
            w_p_sw, w_m_sw = get_rates(i, N, beta, envs[1 - e_idx])

            p_stay_up = (1 - p_sw) * w_p_stay
            p_stay_dn = (1 - p_sw) * w_m_stay
            p_stay_same = (1 - p_sw) * (1 - w_p_stay - w_m_stay)

            p_sw_up = p_sw * w_p_sw
            p_sw_dn = p_sw * w_m_sw
            p_sw_same = p_sw * (1 - w_p_sw - w_m_sw)

            P[row, row] = p_stay_same
            P[row, get_idx(i, 1 - e_idx)] = p_sw_same

            if i > 1:
                P[row, get_idx(i - 1, e_idx)] = p_stay_dn
                P[row, get_idx(i - 1, 1 - e_idx)] = p_sw_dn
            
            if i < N - 1:
                P[row, get_idx(i + 1, e_idx)] = p_stay_up
                P[row, get_idx(i + 1, 1 - e_idx)] = p_sw_up
            else:
                B_phi[row] = p_stay_up + p_sw_up

    I = np.eye(num_states)
    A = I - P
    
    Phi_internal = np.linalg.solve(A, B_phi)
    Theta_internal = np.linalg.solve(A, Phi_internal)
    
    phi_1_plus = Phi_internal[get_idx(1, 0)]   
    phi_1_minus = Phi_internal[get_idx(1, 1)]  
    
    t_1_plus = Theta_internal[get_idx(1, 0)] / phi_1_plus if phi_1_plus > 1e-12 else np.nan
    t_1_minus = Theta_internal[get_idx(1, 1)] / phi_1_minus if phi_1_minus > 1e-12 else np.nan
    
    return t_1_plus, t_1_minus

# ==========================================
# 2. Aproximación de Tasa Efectiva (Ashcroft et al. 2014)
# ==========================================

def solve_ashcroft_effective_prob(p_plus, p_minus_val):
    """ Calcula Probabilidad Efectiva con proporciones actuales """
    P_plus_env = p_plus / (p_plus + p_minus_val)
    P_minus_env = p_minus_val / (p_plus + p_minus_val)
    
    w_p_eff = np.zeros(N)
    w_m_eff = np.zeros(N)
    gamma_eff = np.zeros(N)
    
    for i in range(1, N):
        w_p_1, w_m_1 = get_rates(i, N, beta, 1)
        w_p_m1, w_m_m1 = get_rates(i, N, beta, -1)
        
        w_p_eff[i] = P_plus_env * w_p_1 + P_minus_env * w_p_m1
        w_m_eff[i] = P_plus_env * w_m_1 + P_minus_env * w_m_m1
        gamma_eff[i] = w_m_eff[i] / w_p_eff[i]
        
    prod_gamma = np.ones(N)
    for k in range(1, N):
        prod_gamma[k] = prod_gamma[k-1] * gamma_eff[k]
        
    denominator = 1.0 + np.sum(prod_gamma[1:N])
    phi_1_eff = 1.0 / denominator 
        
    return phi_1_eff

def solve_ashcroft_effective_time(p_plus, p_minus_val):
    """ Calcula Tiempo Efectivo con proporciones correctas """
    # CORREGIDO: Mismas proporciones que en la función de probabilidad
    P_plus_env = p_plus / (p_plus + p_minus_val) 
    P_minus_env = p_minus_val / (p_plus + p_minus_val)     
    
    w_p_eff = np.zeros(N)
    w_m_eff = np.zeros(N)
    gamma_eff = np.zeros(N)
    
    for i in range(1, N):
        w_p_1, w_m_1 = get_rates(i, N, beta, 1)
        w_p_m1, w_m_m1 = get_rates(i, N, beta, -1)
        
        w_p_eff[i] = P_plus_env * w_p_1 + P_minus_env * w_p_m1
        w_m_eff[i] = P_plus_env * w_m_1 + P_minus_env * w_m_m1
        gamma_eff[i] = w_m_eff[i] / w_p_eff[i]
        
    prod_gamma = np.ones(N)
    for k in range(1, N):
        prod_gamma[k] = prod_gamma[k-1] * gamma_eff[k]
        
    denominator = 1.0 + np.sum(prod_gamma[1:N])
    phi_eff = np.zeros(N)
    for i in range(1, N):
        numerator = 1.0 + np.sum(prod_gamma[1:i]) 
        phi_eff[i] = numerator / denominator
        
    t_1_eff = 0.0
    for k in range(1, N):
        inner_sum = 0.0
        for l in range(1, k + 1):
            prod_term = prod_gamma[k] / prod_gamma[l]
            inner_sum += (phi_eff[l] / w_p_eff[l]) * prod_term
        t_1_eff += inner_sum
        
    return t_1_eff

# ==========================================
# 3. Algoritmo de Gillespie y Proceso de Moran
# ==========================================

def gillespie_run(args):
    """ Gillespie SSA for continuous time fixation probability """
    p_plus, p_minus_val, init_env = args
    rng = np.random.default_rng()
    i, env = 1, init_env
    t = 0.0
    while 0 < i < N:
        w_p, w_m = get_rates(i, N, beta, env)
        p_switch = p_plus if env == -1 else p_minus_val

        a0 = w_p + w_m + p_switch
        if a0 <= 0.0:
            break

        t += rng.exponential(1.0 / a0)
        r = rng.random() * a0
        if r < w_p:
            i += 1
        elif r < w_p + w_m:
            i -= 1
        else:
            env *= -1

    return (i == N), t

def discrete_run(args):
    """ Discrete Moran Process for conditional fixation time """
    p_plus, p_minus_val, init_env = args
    i, env, steps = 1, init_env, 0
    
    while 0 < i < N:
        steps += 1
        if random.random() < (p_minus_val if env == 1 else p_plus):
            env *= -1
            
        w_p, w_m = get_rates(i, N, beta, env)
        r = random.random()
        
        if r < w_p: i += 1
        elif r < w_p + w_m: i -= 1
            
    return (i == N, steps)

def simulate_system_prob(p_plus, init_env, runs=4000):
    args = [(p_plus, p_minus, init_env) for _ in range(runs)]
    with mp.Pool() as pool:
        results = pool.map(gillespie_run, args)
    return sum(ok for ok, _ in results) / runs

def simulate_system_time(p_plus, init_env, runs=3000):
    args = [(p_plus, p_minus, init_env) for _ in range(runs)]
    with mp.Pool() as pool:
        results = pool.map(discrete_run, args)
    
    fixation_times = [steps for fixated, steps in results if fixated]
    if len(fixation_times) == 0:
        return np.nan
    return np.mean(fixation_times)

# ==========================================
# 4. Ejecución Principal y Ploteo
# ==========================================
if __name__ == '__main__':
    p_plus_range = np.logspace(-4, 0, 40)
    sim_p_plus = np.logspace(-4, 0, 12)

    # --- 1. Calcular Probabilidades ---
    theory_p, theory_m = [], []
    ashcroft_phi_1 = []
    print("Solving Theory via Matrix Method & Ashcroft (Probabilities)...")
    for p in p_plus_range:
        # Teoría exacta matricial
        tp, tm = solve_prob_matrix(p, p_minus)
        theory_p.append(tp)
        theory_m.append(tm)
        # Teoría Ashcroft (Probabilidades)
        ashcroft_phi_1.append(solve_ashcroft_effective_prob(p, p_minus))

    sim_p, sim_m = [], []
    print("Running Gillespie Simulations (Probabilities)...")
    for p in sim_p_plus:
        print(f"  Simulating p+ = {p:.5f}")
        sim_p.append(simulate_system_prob(p, 1, runs=20000))
        sim_m.append(simulate_system_prob(p, -1, runs=20000))

    # --- 2. Calcular Tiempos Condicionales ---
    theory_t_p, theory_t_m = [], []
    ashcroft_t_1 = []
    print("\nSolving Conditional Time Theory via Matrix Method & Ashcroft...")
    for p in p_plus_range:
        # Teoría exacta matricial
        tp, tm = solve_time_matrix(p, p_minus)
        theory_t_p.append(tp)
        theory_t_m.append(tm)
        # Teoría Ashcroft
        ashcroft_t_1.append(solve_ashcroft_effective_time(p, p_minus))

    sim_t_p, sim_t_m = [], []
    print("Running Simulations (Tracking Time)...")
    for p in sim_p_plus:
        print(f"  Simulating p+ = {p:.5f}")
        sim_t_p.append(simulate_system_time(p, 1, runs=20000))
        sim_t_m.append(simulate_system_time(p, -1, runs=20000))

    # --- 3. Ploteo de ambas figuras juntas ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Probabilidad de fijación
    ax1.plot(p_plus_range, theory_p, 'b-')
    ax1.plot(p_plus_range, theory_m, 'r-')
    # ax1.plot(p_plus_range, ashcroft_phi_1, 'k--') Esto no me parece interesante
    ax1.scatter(sim_p_plus, sim_p, color='blue', marker='+', s=100)
    ax1.scatter(sim_p_plus, sim_m, color='red', marker='o', facecolors='none', s=60)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$p_+$')
    ax1.set_ylabel(r'$\phi_{1,\sigma}$', fontsize=14, rotation=0, labelpad=15)

    # Gráfico 2: Tiempo condicional
    ax2.plot(p_plus_range, theory_t_p, 'b-')
    ax2.plot(p_plus_range, theory_t_m, 'r-')
    # ax2.plot(p_plus_range, ashcroft_t_1, 'k--') Esto no me parece interesante
    ax2.scatter(sim_p_plus, sim_t_p, color='blue', marker='+', s=100)
    ax2.scatter(sim_p_plus, sim_t_m, color='red', marker='o', facecolors='none', s=60)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$p_+$')
    ax2.set_ylabel(r'$t_{1,\sigma}^A$', fontsize=14, rotation=0, labelpad=15)

    plt.tight_layout()
    plt.show()