"""
abm_ec_simulation_v3.py
=======================
Agent-Based Model of Endothelial Cell (EC) Flow-Migration Coupling

Based on Workshop 2.2 framework, restructured to correctly implement the
biophysics described in:

  Edgar et al. (2021) "On the preservation of vessel bifurcations during
  flow-mediated angiogenic remodelling." PLoS Comput Biol 17(2): e1007715.

Key changes from v2 (abm_ec_simulation_v2.py)
----------------------------------------------
1. ECs migrate AGAINST flow (physiologically correct: WSS causes upstream
   polarisation and migration).  v2 had low-probability random downstream
   drift which was biologically backwards.
2. Periodic boundary conditions: cells exiting the inlet re-enter at the
   outlet, conserving the total number of cells throughout the simulation.
3. Three bifurcation rules (BR1, BR3, BR5) implemented at the
   flow-CONVERGENT bifurcation (reunion Node 15), matching the paper.
4. Polarity vectors retained but updated to point AGAINST flow (previously
   pointed WITH flow), consistent with shear-stress driven polarisation.
5. Diagnostic plots match Fig 2 of Edgar et al. (2021).

Network topology — "A-branch" model
-------------------------------------
Inlet → Feeding (segs 0–4) → [Node 5: flow-divergent bifurcation]
                   → Proximal branch (segs 5–14,  10 segments)  →
                   → Distal   branch (segs 20–39, 20 segments)  →  [Node 15: reunion]
                                                                    → Draining (segs 15–19) → Outlet

Cell migration direction (against flow):
  Outlet → Draining → [Node 15: DECISION POINT] → Proximal → Feeding → Inlet (→ Outlet via BC)
                                                 → Distal   ↗

The decision at Node 15 is the "flow-convergent bifurcation" studied in the
paper.  Cells coming from the draining vessel have TWO upstream paths and
must choose one, governed by the bifurcation rule.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Allow importing from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve_for_flow import solve_for_flow
from make_segments import make_segments

# ============================================================
# 1.  PARAMETERS  (Table 1, Edgar et al. 2021)
# ============================================================
Nseg    = 40           # Total vessel segments
n0      = 8            # Initial cells per segment
w       = 5e-6         # EC lateral width  [m]
L_seg   = 10e-6        # Segment length    [m]
mu      = 3.5e-3       # Dynamic viscosity of blood  [Pa·s]
Pin     = 100.0        # Inlet pressure   [Pa]
Pout    = 0.0          # Outlet pressure  [Pa]

# Time-stepping
# Migration speed v = 3 µm/h  →  one step = L_seg/v = 10/3 h ≈ 3.33 h
# 5 days = 120 h  →  Nt = 120 / (10/3) = 36 steps
Nt      = 36
dt_h    = L_seg / 3e-6          # ≈ 3.33 h per time step
dt_days = dt_h / 24.0           # ≈ 0.139 days per step

L        = np.ones(Nseg) * L_seg
segments = make_segments(L)      # For optional network plotting

# ============================================================
# 2.  NETWORK TOPOLOGY — key indices
# ============================================================
# Key segments at the flow-convergent bifurcation (reunion, Node 15)
PROX_LAST   = 14   # Last segment of proximal → ends at Node 15
DIST_LAST   = 39   # Last segment of distal   → ends at Node 15
DRAIN_FIRST = 15   # First seg of draining vessel (starts at Node 15 — the DECISION POINT)


# ============================================================
# 3.  HYDRAULICS  (Hagen–Poiseuille, Eqs 1–3 of paper)
# ============================================================
def compute_conductance(Ncell):
    """
    Compute lumen diameter D, conductance G, and shear-stress factor H
    for every segment.

    Vessel lumen diameter (3-D wrap approximation, Eq 1):
        D = n·w / π

    Conductance (Eq 2):
        G = π D⁴ / (128 µ l_seg)

    Shear-stress factor (from Eq 7):
        τ = H·Q,   H = 32µ / (π D³)

    A minimum diameter of 0.1 µm prevents zero conductance when a
    segment is empty (keeps the linear system invertible).
    """
    # ① vectorised: replaces the original element-wise for loop
    D = np.maximum(Ncell * w / np.pi, 1e-7)
    G = np.pi * D**4 / (128.0 * mu * L_seg)
    H = 32.0  * mu   / (np.pi  * D**3)
    return D, G, H


# ============================================================
# 4.  POLARITY UPDATE  (ECs polarise AGAINST flow)
# ============================================================
# Weight coefficients for polarity realignment
w1 = 0.30   # Persistence (previous direction)
w2 = 0.30   # Flow component (against flow → upstream direction)
w3 = 0.10   # Neighbour alignment
w4 = 0.30   # Random walk

def upstream_direction(seg):
    """
    Return the unit vector pointing UPSTREAM (against flow) for a given
    segment, based on the network geometry defined in make_segments.py.

    Segments 0–4   : feeding vessel, flow goes UP   → upstream = DOWN  [0, -1]
    Segments 5–14  : proximal, flow goes RIGHT       → upstream = LEFT  [-1,  0]
    Segments 15–19 : draining, flow goes DOWN        → upstream = UP    [ 0, +1]
    Segments 20–24 : distal upper, flow goes UP      → upstream = DOWN  [0, -1]
    Segments 25–34 : distal across, flow goes RIGHT  → upstream = LEFT  [-1,  0]
    Segments 35–39 : distal down, flow goes DOWN     → upstream = UP    [ 0, +1]
    """
    if   (0  <= seg <=  4) or (20 <= seg <= 24):  return np.array([ 0., -1.])
    elif (5  <= seg <= 14) or (25 <= seg <= 34):  return np.array([-1.,  0.])
    elif (15 <= seg <= 19) or (35 <= seg <= 39):  return np.array([ 0., +1.])
    return np.array([0., 0.])

def update_polarity(seg_polarity, seg, Q):
    """
    Update the polarity vector of a single cell in `seg`.

    Polarity update (4-component weighted sum):
      new_pol = w1·p_old  +  w2·(−flow_dir)  +  w3·p_neigh  +  w4·rand
    where −flow_dir is the upstream direction (against flow).

    The updated vector is normalised to unit length.
    Note: in the original paper polarity is implicit (all cells move
    upstream), but tracking it provides a useful visualisation of
    collective alignment that emerges over time.
    """
    p_old = np.array(seg_polarity)

    # Upstream direction, corrected for Q sign
    up_dir = upstream_direction(seg)
    if Q[seg] < 0:
        up_dir = -up_dir          # reverse if flow is reversed

    # Random component
    r = np.random.randn(2)
    norm_r = np.linalg.norm(r)
    rand_dir = r / norm_r if norm_r > 0 else np.array([0., 0.])

    # Neighbour direction = same as upstream bias (simplified: no seg_cells passed)
    neigh_dir = up_dir

    new_pol = w1*p_old + w2*up_dir + w3*neigh_dir + w4*rand_dir
    norm_new = np.linalg.norm(new_pol)
    return (new_pol / norm_new) if norm_new > 0 else p_old


# ============================================================
# 5.  BIFURCATION RULES  (Section "EC migration…", Eqs 8–15)
# ============================================================
def apply_br(rule, alpha, tau, Ncell):
    """
    Stochastic bifurcation decision for ONE cell arriving at the
    flow-convergent bifurcation (Node 15).

    The cell migrates AGAINST flow and must choose:
      - Proximal branch  (seg PROX_LAST = 14):  shorter, initially higher τ
      - Distal   branch  (seg DIST_LAST = 39):  longer,  initially lower  τ

    BR1 (Eq 8) — Largest shear stress wins (deterministic)
      Always chooses the branch with higher wall shear stress.
      → Distal branch always regresses (proximal τ > distal τ initially).

    BR3 (Eq 10) — Equal probability  P₁ = P₂ = 0.5
      Uniform random choice between branches.
      → Both branches stabilise at similar diameter.

    BR5 (Eqs 13–15) — Combined shear-stress + cell-number cue
      P_i = α · P_{τi} + (1−α) · P_{ni}
      where P_{τi} = τ_i/(τ₁+τ₂)  and  P_{ni} = n_i/(n₁+n₂).
      Optimal stability around α ≈ 0.45 (Fig 3 of paper).
    """
    τ_p = abs(tau[PROX_LAST]);    τ_d = abs(tau[DIST_LAST])
    n_p = Ncell[PROX_LAST];        n_d = Ncell[DIST_LAST]

    if rule == 1:
        # BR1: deterministic — always follow max τ
        return PROX_LAST if τ_p >= τ_d else DIST_LAST

    elif rule == 3:
        # BR3: fair coin flip
        return PROX_LAST if np.random.rand() < 0.5 else DIST_LAST

    elif rule == 5:
        # BR5: weighted combination (Eq 13)
        τ_tot = τ_p + τ_d;   n_tot = n_p + n_d
        P_τ_p = τ_p / τ_tot if τ_tot > 0 else 0.5
        P_n_p = n_p  / n_tot if n_tot > 0 else 0.5
        P_p   = alpha * P_τ_p + (1.0 - alpha) * P_n_p
        return PROX_LAST if np.random.rand() < P_p else DIST_LAST

    return PROX_LAST   # fallback


# ============================================================
# 6.  MIGRATION STEP  (all cells move one segment upstream)
# ============================================================
def migrate(Ncell, tau, rule, alpha):
    """
    Advance every EC one segment upstream (against flow) simultaneously.

    Migration logic per segment type
    ---------------------------------
    Feeding (0–4):
      Seg 0 → cells EXIT at inlet → PERIODIC BC → re-enter at seg 19
      Segs 1–4 → cells shift one position toward inlet

    Node-5 junction (flow-divergent, migration-convergent):
      Seg  5 (proximal start) → feeds into seg 4 (feeding end)
      Seg 20 (distal   start) → feeds into seg 4  (same junction)
      No decision needed: only one upstream path (the feeding vessel).

    Proximal branch (6–14):
      Segs 6–14 → simple upstream shift (seg s → seg s-1)
      Seg PROX_LAST (14) receives cells from the bifurcation decision below.

    Distal branch (21–39):
      Segs 21–39 → simple upstream shift (seg s → seg s-1)
      Seg DIST_LAST (39) receives cells from the bifurcation decision below.

    Node-15 junction — flow-CONVERGENT bifurcation (THE KEY DECISION):
      Each cell in seg DRAIN_FIRST (15) independently applies the BR rule
      and enters either PROX_LAST (14) or DIST_LAST (39).

    Draining (16–19):
      Segs 16–19 → simple upstream shift (seg s → seg s-1)
      Seg 15 receives cells from seg 16.

    Cell conservation: every Ncell[s] is consumed exactly once and
    redistributed to a unique destination, so sum(new_Ncell) = sum(Ncell).

    Note on intercalation:
      Edgar et al. include a smoothing step ("if cells leaving > incoming
      AND incoming > 0, one cell stays behind") to reduce sharp diameter
      oscillations.  It is omitted here for clarity; its main effect is to
      damp transient fluctuations without changing steady-state behaviour.
    """
    new_N = np.zeros(Nseg)

    # ② numpy slices replace the original for loops (same arithmetic result)
    new_N[19]    += Ncell[0]              # periodic BC: inlet exit → outlet entry
    new_N[0:4]   += Ncell[1:5]           # feeding segs 1-4 → segs 0-3
    new_N[4]     += Ncell[5] + Ncell[20] # ③ Node 5: both branch starts → seg 4
    new_N[5:14]  += Ncell[6:15]          # proximal interior: segs 6-14 → segs 5-13
    new_N[20:39] += Ncell[21:40]         # distal interior:   segs 21-39 → segs 20-38
    new_N[15:19] += Ncell[16:20]         # draining interior: segs 16-19 → segs 15-18

    # --- Node 15 bifurcation: each cell in DRAIN_FIRST chooses a branch ---
    for _ in range(int(round(Ncell[DRAIN_FIRST]))):
        new_N[apply_br(rule, alpha, tau, Ncell)] += 1

    return new_N


# ============================================================
# 7.  SINGLE SIMULATION RUN
# ============================================================
def run_sim(rule, alpha=0.45, seed=0):  # ④ removed unused track_polarity parameter
    """
    Run the ABM for Nt time steps with the specified bifurcation rule.

    Parameters
    ----------
    rule  : int   — bifurcation rule (1, 3, or 5)
    alpha : float — BR5 weight (ignored for BR1 and BR3)
    seed  : int   — random seed for reproducibility

    Returns
    -------
    prox_D  : ndarray (Nt+1,) — mean proximal branch diameter [µm]
    dist_D  : ndarray (Nt+1,) — mean distal   branch diameter [µm]
    lost    : bool             — True if bifurcation was declared lost
    t_lost  : int or None      — step at which bifurcation was lost
    Ncell   : ndarray (Nseg,) — final cell count per segment
    """
    np.random.seed(seed)

    # --- Initialise ---
    Ncell = np.ones(Nseg) * n0
    D, G, H = compute_conductance(Ncell)
    _, Q, tau = solve_for_flow(G, Pin, Pout, H)

    # Initialise polarity vectors: random unit vectors
    polarity = {}
    for s in range(Nseg):
        polarity[s] = [np.array([0., -1.]) for _ in range(n0)]  # start pointing upstream

    prox_D = [np.mean(D[5:15]) * 1e6]
    dist_D = [np.mean(D[20:40]) * 1e6]
    lost, t_lost = False, None

    for t in range(Nt):
        # --- Polarity update (against flow, weighted) ---
        for s in range(Nseg):
            n_s = int(round(Ncell[s]))
            new_pols = []
            for i in range(n_s):
                p_old = polarity[s][i] if i < len(polarity[s]) else np.array([0., -1.])
                new_pols.append(update_polarity(p_old, s, Q))
            polarity[s] = new_pols

        # --- Cell migration (all cells move one segment upstream) ---
        Ncell = migrate(Ncell, tau, rule, alpha)

        # Redistribute polarity lists to match new Ncell
        # (simplified: reset polarity of cells in new segments based on segment direction)
        new_polarity = {}
        for s in range(Nseg):
            n_s = int(round(Ncell[s]))
            up = upstream_direction(s)
            new_polarity[s] = []
            for _ in range(n_s):
                r = np.random.randn(2) * 0.1
                p = up + r
                nrm = np.linalg.norm(p)
                new_polarity[s].append(p / nrm if nrm > 0 else up)
        polarity = new_polarity

        # --- Update flow ---
        D, G, H = compute_conductance(Ncell)
        _, Q, tau = solve_for_flow(G, Pin, Pout, H)

        # --- Record ---
        prox_D.append(np.mean(D[5:15]) * 1e6)
        dist_D.append(np.mean(D[20:40]) * 1e6)

        # --- Detect bifurcation loss ---
        # Paper definition: cell count in one of the two segments at the
        # flow-convergent bifurcation drops to zero (Edgar et al., 2021).
        if not lost and (Ncell[PROX_LAST] == 0 or Ncell[DIST_LAST] == 0):
            lost, t_lost = True, t

    return np.array(prox_D), np.array(dist_D), lost, t_lost, Ncell


# ============================================================
# 8.  BR5 PROBABILITY TRACKER (for oscillation analysis)
# ============================================================
def run_sim_with_probabilities(alpha=0.45, seed=0):
    """
    Run BR5 and record the branch probabilities P_tau and P_n over time,
    enabling reproduction of Fig 4 of Edgar et al. (2021).
    """
    np.random.seed(seed)
    Ncell = np.ones(Nseg) * n0
    D, G, H = compute_conductance(Ncell)
    _, Q, tau = solve_for_flow(G, Pin, Pout, H)

    prox_D, dist_D = [np.mean(D[5:15])*1e6], [np.mean(D[20:40])*1e6]
    P_total, P_tau_hist, P_n_hist = [], [], []

    for _ in range(Nt):
        τ_p = abs(tau[PROX_LAST]);  τ_d = abs(tau[DIST_LAST])
        n_p = Ncell[PROX_LAST];      n_d = Ncell[DIST_LAST]

        τ_tot = τ_p + τ_d;  n_tot = n_p + n_d
        P_τ_p = τ_p/τ_tot if τ_tot>0 else 0.5
        P_n_p = n_p/n_tot  if n_tot>0 else 0.5
        P_p   = alpha*P_τ_p + (1-alpha)*P_n_p

        P_total.append(P_p)
        P_tau_hist.append(P_τ_p)
        P_n_hist.append(P_n_p)

        Ncell = migrate(Ncell, tau, 5, alpha)
        D, G, H = compute_conductance(Ncell)
        _, Q, tau = solve_for_flow(G, Pin, Pout, H)

        prox_D.append(np.mean(D[5:15])*1e6)
        dist_D.append(np.mean(D[20:40])*1e6)

    return (np.array(prox_D), np.array(dist_D),
            np.array(P_total), np.array(P_tau_hist), np.array(P_n_hist))


# ============================================================
# 9.  PLOTTING
# ============================================================
def _plot_multi_seed(ax, t_ax, rule, alpha, n_runs):
    """⑤ Shared helper: run n_runs seeds and plot mean ± SD bands."""
    all_p, all_d, n_lost = [], [], 0
    for seed in range(n_runs):
        pD, dD, lost, _, _ = run_sim(rule=rule, alpha=alpha, seed=seed)
        all_p.append(pD); all_d.append(dD)
        if lost: n_lost += 1
    AP, AD = np.array(all_p), np.array(all_d)
    for arr, col, lbl in [(AP, 'b', 'Proximal branch'), (AD, 'r', 'Distal branch')]:
        m, s = arr.mean(0), arr.std(0)
        ax.plot(t_ax, m, f'{col}-', lw=2.5, label=lbl)
        ax.fill_between(t_ax, m-s, m+s, alpha=0.2, color=col)
    return AP, AD, n_lost

def _ax_style(ax):
    ax.set_xlabel('Time (days)'); ax.set_ylabel('Mean diameter (µm)')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

def _status_box(ax, txt, color):
    ax.text(0.97, 0.95, txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color=color,
            bbox=dict(boxstyle='round', fc='white', ec=color, alpha=0.9))


def make_comparison_figure(n_runs=10, alpha_br5=0.45, out_path=None):
    """
    Produce a 3-panel comparison figure (BR1 / BR3 / BR5) matching the
    style of Fig 2 in Edgar et al. (2021).

    Returns lists of prox/dist diameter arrays for further analysis.
    """
    t_ax = np.arange(Nt+1) * dt_days

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle(
        'Endothelial Cell Migration ABM — Bifurcation Rules Comparison\n'
        'Reproducing Fig 2, Edgar et al. (2021)',
        fontsize=13, y=1.01
    )

    results = {}

    # -------- BR1 --------
    ax = axes[0]
    pD, dD, lost, tl, Nc = run_sim(rule=1, seed=0)
    ax.plot(t_ax, pD, 'b-', lw=2.5, label='Proximal branch')
    ax.plot(t_ax, dD, 'r-', lw=2.5, label='Distal branch')
    if lost and tl is not None:
        ax.axvline(tl*dt_days, color='gray', ls='--', lw=1.5)
    ax.set_title('BR1: Max shear stress\n'
                 '(cells always enter higher-τ branch)', fontsize=11)
    _status_box(ax, 'BIFURCATION LOST' if lost else 'Stable', 'red' if lost else 'green')
    _ax_style(ax)
    results['BR1'] = (pD, dD, lost)

    # -------- BR3 and BR5 (⑤ unified via shared helper) --------
    for ax, rule, alpha, title, key in [
        (axes[1], 3, 0.5,      f'BR3: Equal probability P=0.5\n(mean ± SD over {n_runs} seeds)', 'BR3'),
        (axes[2], 5, alpha_br5, f'BR5: Combined cues (α={alpha_br5})\n(mean ± SD over {n_runs} seeds)', 'BR5'),
    ]:
        AP, AD, n_lost = _plot_multi_seed(ax, t_ax, rule, alpha, n_runs)
        n_stable = n_runs - n_lost
        ax.set_title(title, fontsize=11)
        clr = 'green' if n_stable >= n_runs * 0.6 else 'orange'
        _status_box(ax, f'{n_stable}/{n_runs} stable', clr)
        _ax_style(ax)
        results[key] = (AP, AD, n_lost)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'  Saved → {out_path}')
    plt.show()
    plt.close()
    return results


def make_probability_figure(alpha=0.45, seed=3, out_path=None):
    """
    For one BR5 run, plot the branch probability and its shear/cell-number
    decomposition over time (style of Fig 4, Edgar et al. 2021).
    """
    prox_D, dist_D, P_tot, P_tau, P_n = run_sim_with_probabilities(alpha, seed)
    t_ax  = np.arange(Nt+1) * dt_days
    t_mid = np.arange(Nt)   * dt_days + dt_days/2   # midpoints for probabilities

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f'BR5 (α={alpha}) — Probability Decomposition\n'
                 f'Reproducing Fig 4, Edgar et al. (2021)', fontsize=13)

    # Panel A: mean diameter
    ax = axes[0]
    ax.plot(t_ax, prox_D, 'b-', lw=2, label='Proximal branch')
    ax.plot(t_ax, dist_D, 'r-', lw=2, label='Distal branch')
    ax.set_ylabel('Mean diameter (µm)'); ax.set_xlabel('Time (days)')
    ax.set_title('Branch diameter over time'); ax.legend(); ax.grid(True, alpha=0.3)

    # Panel B: probability decomposition
    ax = axes[1]
    ax.plot(t_mid, P_tot, 'k-',  lw=2.5, label='P_prox (total)')
    ax.plot(t_mid, P_tau, 'b--', lw=1.5, label='P_τ (shear stress component)')
    ax.plot(t_mid, P_n,   'r:',  lw=1.5, label='P_n (cell number component)')
    ax.axhline(0.5, color='gray', ls=':', lw=1)
    ax.set_ylim(0, 1); ax.set_ylabel('Probability of proximal branch')
    ax.set_xlabel('Time (days)'); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_title('Shear-stress vs. cell-number probability (competitive oscillations)')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'  Saved → {out_path}')
    plt.show()
    plt.close()


# ============================================================
# 10.  MAIN — run and report
# ============================================================
if __name__ == '__main__':
    N_RUNS    = 10       # paper uses 1000; 10 is sufficient for illustration
    ALPHA_BR5 = 0.45     # peak stability value from paper (Fig 3)
    OUT_DIR   = './results'
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Initial flow diagnostics ---
    D0, G0, H0 = compute_conductance(np.ones(Nseg)*n0)
    _, _, tau0 = solve_for_flow(G0, Pin, Pout, H0)  # ⑥ Q0 was unused, replaced with _
    print('='*55)
    print(' EC Flow-Migration ABM — v3')
    print('='*55)
    print(f' Segments: {Nseg},  n0={n0} cells/seg,  w={w*1e6:.0f} µm')
    print(f' Segment length: {L_seg*1e6:.0f} µm')
    print(f' Time step: {dt_h:.2f} h  →  Nt={Nt} steps ({Nt*dt_days:.1f} days)')
    print()
    print(f' Initial shear stress:')
    print(f'   Proximal (seg {PROX_LAST}): {tau0[PROX_LAST]:.3f} Pa')
    print(f'   Distal   (seg {DIST_LAST}): {tau0[DIST_LAST]:.3f} Pa')
    print(f'   Ratio  τ_prox/τ_dist  =  {tau0[PROX_LAST]/tau0[DIST_LAST]:.2f}×')
    print()
    print(f' Initial mean diameter: {np.mean(D0[5:15])*1e6:.2f} µm (all branches equal)')
    print()

    # --- Run simulations ---
    print(' Running BR1, BR3, BR5 comparison ...')
    results = make_comparison_figure(
        n_runs=N_RUNS,
        alpha_br5=ALPHA_BR5,
        out_path=f'{OUT_DIR}/fig_bifurcation_rules.png'
    )

    # Summary statistics
    print()
    print(' Summary:')
    _, _, br1_lost = results['BR1']
    _, _, br3_lost = results['BR3']
    _, _, br5_lost = results['BR5']
    print(f'   BR1 — Bifurcation lost: {"YES" if br1_lost else "no"}')
    print(f'   BR3 — {N_RUNS - br3_lost}/{N_RUNS} stable ({100*(1-br3_lost/N_RUNS):.0f}%)')
    print(f'   BR5 (α={ALPHA_BR5}) — {N_RUNS - br5_lost}/{N_RUNS} stable ({100*(1-br5_lost/N_RUNS):.0f}%)')
    print()
    print(' Paper reference values (1000 seeds):')
    print('   BR1 — 100% lost (distal always regresses)')
    print('   BR3 — ~100% stable, equal diameters')
    print('   BR5 (α=0.45) — ~77% stable, proximal > distal')

    # --- BR5 probability figure ---
    print()
    print(' Plotting BR5 probability decomposition ...')
    make_probability_figure(
        alpha=ALPHA_BR5,
        seed=3,
        out_path=f'{OUT_DIR}/fig_br5_probabilities.png'
    )

    print()
    print(' All done.')
