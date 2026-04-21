import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

t0 = time.perf_counter()

N       = 50
beta    = 1.0
M_SIM   = 5_000   # todas las trayectorias a la vez

mu_sim     = np.linspace(-3, 10, 18)
mu_th      = np.linspace(-3, 10, 300)
mu_grid    = np.linspace(-3, 10, 70)
sigma_grid = np.linspace(0.1, 8, 70)

p1_list    = [0.25, 0.50, 0.75]
sigma_cuts = [1, 4]
colors_cut = ['tab:red', 'tab:blue']
markers    = ['o', 's']
titles     = [r'Left-skewed  ($p_1=0.25$)',
              r'Symmetric  ($p_1=0.50$)',
              r'Right-skewed  ($p_1=0.75$)']

phi_neutral = 1.0 / N
np.random.seed(42)

def g_plus(a):
    return 1.0 / (1.0 + np.exp(-beta * np.asarray(a, dtype=float)))

def get_payoffs(mu, sigma, p1):
    p2 = 1.0 - p1
    return mu - sigma*np.sqrt(p2/p1), mu + sigma*np.sqrt(p1/p2)

def phi_theory_scalar(mu, sigma, p1):
    a1, a2 = get_payoffs(mu, sigma, p1)
    gp = p1*g_plus(a1) + (1-p1)*g_plus(a2)
    gamma = (1.0 - gp) / gp
    if abs(gamma - 1.0) < 1e-12:
        return phi_neutral
    return float((1.0 - gamma) / (1.0 - gamma**N))

def phi_theory_2d(mu_arr, sigma_arr, p1):
    MU, SIG = np.meshgrid(mu_arr, sigma_arr)
    A1 = MU - SIG*np.sqrt((1-p1)/p1)
    A2 = MU + SIG*np.sqrt(p1/(1-p1))
    gp  = p1*g_plus(A1) + (1-p1)*g_plus(A2)
    GAM = (1.0 - gp) / gp
    return np.where(np.abs(GAM-1) < 1e-12,
                    phi_neutral,
                    (1.0 - GAM) / (1.0 - GAM**N))

def simulate_phi(mu_arr, sigma, p1, verbose=False):
    """Fully vectorised — no batch loop."""
    p2 = 1.0 - p1
    out = []
    for k, mu in enumerate(mu_arr):
        if verbose:
            print(f"  mu={mu:+.2f} ({k+1}/{len(mu_arr)})", end="\r")
        a1, a2 = get_payoffs(mu, sigma, p1)
        gp = float(p1*g_plus(a1) + p2*g_plus(a2))
        gm = 1.0 - gp

        pop   = np.ones(M_SIM, dtype=np.int16)
        alive = np.ones(M_SIM, dtype=bool)
        fixed = np.zeros(M_SIM, dtype=bool)

        while alive.any():
            idx  = np.where(alive)[0]
            iv   = pop[idx].astype(np.float32)
            base = iv * (N - iv) / (N*N)
            pp   = base * gp
            pm   = base * gm
            r    = np.random.random(len(idx)).astype(np.float32)
            di   = np.zeros(len(idx), dtype=np.int16)
            di[r < pp]                   =  1
            di[(r >= pp) & (r < pp+pm)]  = -1
            pop[idx] += di
            cur  = pop[idx]
            fm   = cur == N
            done = fm | (cur == 0)
            fixed[idx[fm]]   = True
            alive[idx[done]] = False

        out.append(float(fixed.mean()))
    if verbose: print()
    return np.array(out)

# ── Figure ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
cmap_phi = plt.cm.RdBu_r
norm_phi = mcolors.TwoSlopeNorm(vmin=0, vcenter=phi_neutral, vmax=1)

for ax, p1, title in zip(axes, p1_list, titles):
    print(f"\n── p1={p1} ──────────────")
    for sigma, col, mrk in zip(sigma_cuts, colors_cut, markers):
        phi_th = np.array([phi_theory_scalar(mu, sigma, p1) for mu in mu_th])
        ax.plot(mu_th, phi_th, color=col, lw=2.0, label=fr'$\sigma={sigma}$')
        print(f"  sigma={sigma}  MC...")
        phi_sim = simulate_phi(mu_sim, sigma, p1, verbose=True)
        ax.scatter(mu_sim, phi_sim, color=col, marker=mrk,
                   s=22, edgecolors='k', linewidths=0.4, zorder=5)

    #ax.axhline(phi_neutral, color='gray', ls='--', lw=1.0, alpha=0.7)
    ax.set_title(title, fontsize=11, pad=5)
    ax.set_xlabel(r'Mean payoff $\mu$', fontsize=11)
    ax.set_xlim([-3, 10]);  ax.set_ylim([-0.02, 1.05])
    ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
    ax.tick_params(labelsize=9);  ax.grid(alpha=0.22)

    # ── Inset heatmap ────────────────────────────────
    mu_grid_in    = np.linspace(-8, 8, 70)     
    sigma_grid_in = np.linspace(0.0, 7, 70)   

    ax_in = ax.inset_axes([0.52, 0.04, 0.46, 0.52])
    Z = phi_theory_2d(mu_grid_in, sigma_grid_in, p1) 
    ax_in.pcolormesh(mu_grid_in, sigma_grid_in, Z, cmap=cmap_phi, norm=norm_phi,
                     shading='auto', rasterized=True)
    ax_in.contour(mu_grid_in, sigma_grid_in, Z, levels=[phi_neutral],
                  colors='white', linewidths=1.2, linestyles='-')
    for sigma, col in zip(sigma_cuts, colors_cut):
        ax_in.axhline(sigma, color=col, ls='--', lw=1.3)
    ax_in.xaxis.set_label_position('top')          
    ax_in.xaxis.tick_top()                         
    ax_in.set_xlabel(r'$\mu$', fontsize=8, labelpad=2)
    ax_in.set_ylabel(r'$\sigma$', fontsize=8, labelpad=1)
    ax_in.tick_params(labelsize=7)
    ax_in.set_xlim([-8, 8])                      
    ax_in.set_ylim([0, 7])                        

axes[0].set_ylabel(r'Fixation probability $\phi_1$', fontsize=11)

cbar_ax = fig.add_axes([0.92, 0.15, 0.014, 0.68])
sm = plt.cm.ScalarMappable(cmap=cmap_phi, norm=norm_phi)
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label(r'$\phi_1$', fontsize=10)
cb.ax.tick_params(labelsize=8)

out = r'\proyecto_t4_fast_switching.png'
plt.savefig(out, dpi=160, bbox_inches='tight')
plt.close()
print(f"\nGuardado -> {out}")
print(f"Tiempo total: {time.perf_counter()-t0:.1f} s")
