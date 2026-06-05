### Trayectorias de Gillespie para el apartado de dinamica de fijacion
### en entornos fluctuantes (juego de Coexistencia / Coordinacion).
###
### Genera UNA figura con tres paneles en horizontal (proyecto_t3_trayectorias.png):
###   (izquierda) conmutacion lenta  : mesetas largas pegadas a cada entorno.
###   (centro)    metaestabilidad    : regimen de resonancia (p+ = p- = 0.01);
###               acercamientos al atractor i=0 y rescates por conmutacion.
###   (derecha)   conmutacion rapida : el entorno se promedia y la trayectoria
###               mantiene un equilibrio mixto.
### Las tres parten de un unico mutante (i0=1) y conservan su propio eje
### temporal (cada trayectoria dura lo suyo).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==========================================
# Parameters
# ==========================================
N = 50           # Population size
beta = 1.0       # Intensity of selection
p_minus = 0.01   # Switching rate from sigma=+1 to sigma=-1

# Game payoff matrix per environment:
#   Pi_sigma = [[1, 1 + sigma*b], [1 + sigma*c, 1]],  b=0.5, c=0.9
# sigma=+1 -> Coexistence (favours the minority strategy, interior mixed eq.)
# sigma=-1 -> Coordination (drives the system to the boundaries 0 or N)
B_PAR, C_PAR = 0.5, 0.9

ENV_LABEL = {1: "Coexistencia ($\\sigma=+1$)", -1: "Coordinaci\u00f3n ($\\sigma=-1$)"}
COL_PLUS, COL_MINUS = "#cfe8ff", "#ffd9d9"

def get_payoffs(i, N, sigma):
    a, d = 1.0, 1.0
    b = 1.0 + sigma * B_PAR
    c = 1.0 + sigma * C_PAR
    pi_A = ((i - 1) * a + (N - i) * b) / (N - 1)
    pi_B = (i * c + (N - i - 1) * d) / (N - 1)
    return pi_A, pi_B

def get_rates(i, N, beta, sigma):
    """Fermi birth-death rates for the current state and environment."""
    pi_A, pi_B = get_payoffs(i, N, sigma)
    f_A, f_B = np.exp(beta * pi_A), np.exp(beta * pi_B)
    f_bar = (i * f_A + (N - i) * f_B) / N
    w_plus = (i * (N - i) / N**2) * (f_A / f_bar)
    w_minus = (i * (N - i) / N**2) * (f_B / f_bar)
    return w_plus, w_minus

# ==========================================
# Trayectoria de Gillespie
# ==========================================
def gillespie_trajectory(p_plus, init_env, i0=1, t_max=None, max_events=500000,
                         seed=None, stop_on_absorption=True):
    """
    One SSA trajectory of the switching-environment birth-death process.

    Events and rates:
        birth        w_plus   (i -> i+1)
        death        w_minus  (i -> i-1)
        env. switch  p_sigma  (sigma -> -sigma)
    Total propensity a0 = w_plus + w_minus + p_sigma.

    p_sigma is the rate of LEAVING the current environment: leave sigma=+1 at
    rate p_minus, leave sigma=-1 at rate p_plus. Note i=0 and i=N are absorbing
    for the demographic process (w_plus, w_minus ~ i(N-i) vanish there), so we
    stop on absorption.

    Returns dict with continuous-time arrays t, i, sigma (step functions).
    """
    rng = np.random.default_rng(seed)
    p_leave = {1: p_minus, -1: p_plus}

    i, env, t = i0, init_env, 0.0
    ts, iss, sigs = [0.0], [i], [env]

    n_events = 0
    while n_events < max_events:
        if stop_on_absorption and (i == 0 or i == N):
            break
        if t_max is not None and t >= t_max:
            break

        w_p, w_m = get_rates(i, N, beta, env)
        p_switch = p_leave[env]
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

        ts.append(t); iss.append(i); sigs.append(env)
        n_events += 1

    return {"t": np.array(ts), "i": np.array(iss), "sigma": np.array(sigs)}


def _shade_environments(ax, t, sigma):
    """Shade the time axis by environmental state (grouped step function)."""
    k = 0
    n = len(t)
    while k < n - 1:
        j = k
        while j < n - 1 and sigma[j] == sigma[k]:
            j += 1
        c = COL_PLUS if sigma[k] == 1 else COL_MINUS
        ax.axvspan(t[k], t[j], color=c, linewidth=0, zorder=0)
        k = j


def _env_legend(ax, loc="upper left"):
    handles = [Patch(facecolor=COL_PLUS, label=ENV_LABEL[1]),
               Patch(facecolor=COL_MINUS, label=ENV_LABEL[-1])]
    ax.legend(handles=handles, loc=loc, framealpha=0.9, fontsize=8)


def _reference_lines(ax):
    """Dotted guides at x = 0, 1 and dashed at x = 0.5."""
    for y in (0.0, 1.0):
        ax.axhline(y, color="0.4", ls=":", lw=1)
    ax.axhline(0.5, color="0.55", ls="--", lw=1)


def _plot_trajectory(ax, traj, lw=1.2):
    _shade_environments(ax, traj["t"], traj["sigma"])
    ax.step(traj["t"], traj["i"] / N, where="post", color="black", lw=lw)
    _reference_lines(ax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, traj["t"][-1])
    ax.set_ylabel("$x = i/N$")


# ==========================================
# Plotteado
# ==========================================
def figure_combined():
    # --- Top: slow switching (long residence per environment) ---
    p_slow = 0.004
    tr_slow = gillespie_trajectory(p_slow, init_env=1, i0=1, t_max=40000.0,
                                   seed=29, stop_on_absorption=True)

    # --- Middle: metastability at stochastic resonance p+ = p- = 0.01 ---
    p_res = 0.01
    tr_meta = gillespie_trajectory(p_res, init_env=1, i0=1, t_max=30000.0,
                                   seed=578, stop_on_absorption=True)

    # --- Bottom: fast switching, mixed equilibrium (cut before absorption) ---
    p_fast = 1.0
    tr_fast = gillespie_trajectory(p_fast, init_env=1, i0=1, t_max=22000.0,
                                   seed=17, stop_on_absorption=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    _plot_trajectory(axes[0], tr_slow)
    axes[0].set_title("($p_+ = %.3f$)" % p_slow)
    _env_legend(axes[0], loc="upper left")

    _plot_trajectory(axes[1], tr_meta)
    axes[1].set_title("($p_+ = p_- = %.2f$)" % p_res)

    _plot_trajectory(axes[2], tr_fast)
    axes[2].set_title("($p_+ = %.1f$)" % p_fast)

    # eje y solo en el panel izquierdo; eje x en los tres
    for a in axes:
        a.set_xlabel("Tiempo ($t$)")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    fig.tight_layout()
    fig.savefig("proyecto_t3_trayectorias.png", dpi=200)
    return fig


if __name__ == "__main__":
    figure_combined()
    plt.show()