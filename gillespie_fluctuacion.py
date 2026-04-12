### Script for the third task of the SC project ###

import numpy as np

def gillespie_fluctuating_environment(N, beta, matrix_1, matrix_2, p_plus, p_minus, initial_A=1, runs=1000):
    fixation_count = 0
    fixation_times = []
    
    for _ in range(runs):
        i = initial_A
        t = 0.0
        env = 1 # Start in environment 1
        
        while 0 < i < N:
            # Current payoff matrix
            matrix = matrix_1 if env == 1 else matrix_2
            pi_A = (i-1)/N * matrix[0,0] + (N-i)/N * matrix[0,1]
            pi_B = i/N * matrix[1,0] + (N-i-1)/N * matrix[1,1]
            delta_pi = pi_A - pi_B
            
            # Transition rates
            T_plus = (i/N) * ((N-i)/N) / (1 + np.exp(-beta * delta_pi))
            T_minus = (i/N) * ((N-i)/N) / (1 + np.exp(beta * delta_pi))
            
            # Total event rate (Birth/Death + Environment Switch)
            # Assuming env switches at rates related to p_plus / p_minus
            R_env = p_plus if env == -1 else p_minus
            R_total = T_plus + T_minus + R_env
            
            # Time to next event
            tau = np.random.exponential(1 / R_total)
            t += tau
            
            # Determine which event occurred
            r = np.random.uniform(0, R_total)
            if r < T_plus:
                i += 1
            elif r < T_plus + T_minus:
                i -= 1
            else:
                env *= -1 # Environment flips
                
        if i == N:
            fixation_count += 1
            fixation_times.append(t)
            
    phi_1 = fixation_count / runs
    mean_t_1 = np.mean(fixation_times) if fixation_times else 0
    return phi_1, mean_t_1
