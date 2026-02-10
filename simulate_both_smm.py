"""
simulate_both_smm.py
  Load calibrated parameters from .txt files and simulate from both
  FDM and Hedging SV-X models.

  Model:
    r_t = alpha + beta*m_t + gamma'*s_t + exp(0.5*h_t)*eps_t
          + J_t*Z_t - p_{j,t}*mu_j                  [jump-compensated]
    eps_t ~ Hansen_skewt(nu, lambda)
    xi_t  ~ N(0,1), independent of eps_t
    h_t = omega + phi*(h_{t-1}-omega) + delta_v*v_t
          + sigma_h*(rho*eps_{t-1} + sqrt(1-rho^2)*xi_t)
    p_{j,t} = sigmoid(psi_0 + psi'*v_t)
    Z_t ~ Exp(mu_j)
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import skew, kurtosis


# ======== Helper functions ========

def load_smm_params(filename):
    """Read calibrated SMM parameters from a text file."""
    kv = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            key = tokens[0]
            vals = np.array([float(x) for x in tokens[1:]])
            kv[key] = vals

    # Build param_hat dict
    p = {
        'alpha':   kv['alpha'].item(),
        'beta':    kv['beta'].item(),
        'gamma':   kv['gamma'],
        'omega':   kv['omega'].item(),
        'phi':     kv['phi'].item(),
        'sigmah':  kv['sigmah'].item(),
        'delta_v': kv['delta_v'].item(),
        'nu':      kv['nu'].item(),
        'lambda':  kv['lambda'].item(),
        'rho':     kv['rho'].item(),
        'mu_j':    kv['mu_j'].item(),
        'psi0':    kv['psi0'].item(),
        'psi':     kv['psi'],
    }

    results = {
        'param_hat': p,
        'S_mean':    kv['S_mean'],
        'S_std':     kv['S_std'],
        'V_mean':    kv['V_mean'],
        'V_std':     kv['V_std'],
    }
    print(f'Parameters loaded from {filename}')
    return results


def simulate_model(results, m, S, V_jump, Tsim, Nsim, rng):
    """Simulate return paths from a calibrated SV-X model."""
    p  = results['param_hat']
    Sz = (S - results['S_mean']) / results['S_std']
    Vz = (V_jump - results['V_mean']) / results['V_std']
    K  = Sz.shape[1]
    Kv = Vz.shape[1] if Vz.ndim > 1 else 1

    # Ensure Vz is 2-D
    if Vz.ndim == 1:
        Vz = Vz[:, np.newaxis]

    # Conditional mean (OLS, fixed)
    mu_cond = p['alpha'] + p['beta'] * m + (Sz @ p['gamma'] if K > 0 else 0.0)

    # Time-varying jump probability
    pj_t = 1.0 / (1.0 + np.exp(-(p['psi0'] + (Vz @ p['psi'] if Kv > 0 else 0.0))))
    pj_t = pj_t.ravel()

    # Hansen's skewed-t constants
    nu  = p['nu']
    lam = p['lambda']
    logc  = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    c_skt = np.exp(logc)
    a_skt = 4 * lam * c_skt * (nu - 2) / (nu - 1)
    b_skt = np.sqrt(max(1 + 3 * lam**2 - a_skt**2, 1e-10))

    # Sample from Hansen's skewed-t (continuous nu via gamma draw)
    Z_base = rng.standard_normal((Tsim, Nsim))
    chi2_draws = 2.0 * rng.gamma(nu / 2, 1.0, size=(Tsim, Nsim))
    V_t = Z_base / np.sqrt(chi2_draws / nu)
    V_s = V_t * np.sqrt((nu - 2) / nu)
    U_bern = (rng.random((Tsim, Nsim)) < (1 + lam) / 2)
    W = np.abs(V_s) * ((1 + lam) * U_bern - (1 - lam) * (~U_bern))
    eps_sim = (W - a_skt) / b_skt

    # Simulate h paths with leverage (stationary initialization)
    xi = rng.standard_normal((Tsim, Nsim))
    h_sim = np.zeros((Tsim, Nsim))
    sqrt_1mrho2 = np.sqrt(max(1 - p['rho']**2, 0.0))

    h_sim[0, :] = (p['omega']
                    + p['delta_v'] * Vz[0, :]
                    + p['sigmah'] / np.sqrt(max(1 - p['phi']**2, 1e-6)) * xi[0, :])
    for t in range(1, Tsim):
        h_sim[t, :] = (p['omega']
                        + p['phi'] * (h_sim[t-1, :] - p['omega'])
                        + p['delta_v'] * Vz[t, :]
                        + p['sigmah'] * (p['rho'] * eps_sim[t-1, :] + sqrt_1mrho2 * xi[t, :]))

    # Jump components
    J_sim = (rng.random((Tsim, Nsim)) < pj_t[:, np.newaxis])
    Z_sim = (-p['mu_j'] * np.log(rng.random((Tsim, Nsim)))) * J_sim

    # Returns (jump-compensated)
    r_sim = (mu_cond[:, np.newaxis]
             + np.exp(0.5 * h_sim) * eps_sim
             + Z_sim
             - (pj_t * p['mu_j'])[:, np.newaxis])

    return r_sim, h_sim


def compute_moments(r_sim, m_dm, var_m_scalar, idx_down, m_down_dm, var_m_down):
    """Compute 9 moments from simulated returns."""
    beta_sim      = (m_dm @ r_sim) / var_m_scalar
    beta_down_sim = (m_down_dm @ r_sim[idx_down, :]) / var_m_down

    sim_moments = np.array([
        np.mean(r_sim, axis=0),
        12 * np.mean(r_sim, axis=0),
        np.var(r_sim, axis=0, ddof=0),
        np.std(r_sim, axis=0, ddof=0),
        np.sqrt(12) * np.std(r_sim, axis=0, ddof=0),
        skew(r_sim, axis=0, bias=True),
        kurtosis(r_sim, axis=0, fisher=True, bias=True),
        beta_sim,
        beta_down_sim,
    ])
    return np.mean(sim_moments, axis=1)


# ======== Main ========

def main():
    data_file = 'hedging_states_returns.xlsx'

    # ---- Load data ----
    ret = pd.read_excel(data_file, sheet_name='Portfolio_returns')
    r_fdm = ret.iloc[:, 1].values.astype(float)   # Column B: FDM
    r_hdg = ret.iloc[:, 2].values.astype(float)   # Column C: Hedging
    dates = pd.to_datetime(ret.iloc[:, 0])

    st = pd.read_excel(data_file, sheet_name='States')
    m = st.iloc[:, 2].values.astype(float)         # Column C: STOCK_EXCESS_RETURN
    S = np.column_stack([st.iloc[:, 1].values] +
                        [st.iloc[:, c].values for c in range(3, 16)])  # Columns B, D-P

    vol_tab = pd.read_excel('volatilities.xlsx')
    vol_d = pd.to_datetime(vol_tab.iloc[:, 0])

    # Intersect dates
    dates_set = {d: i for i, d in enumerate(dates)}
    vol_d_set = {d: i for i, d in enumerate(vol_d)}
    common = sorted(set(dates) & set(vol_d))
    i_ret = np.array([dates_set[d] for d in common])
    i_vol = np.array([vol_d_set[d] for d in common])

    V = np.full((len(dates), 4), np.nan)
    V[i_ret, :] = vol_tab.iloc[i_vol, 1:5].values.astype(float)
    S = np.column_stack([S, V])   # 14 + 4 = 18 state variables

    # Keep only data up to 12/31/2025
    idx = dates <= pd.Timestamp('2025-12-31')
    r_fdm = r_fdm[idx];  r_hdg = r_hdg[idx]
    m = m[idx];  S = S[idx, :]
    V_jump = S[:, -4]  # equity vol only

    moments_tab = pd.read_excel(data_file, sheet_name='Moments')
    target_fdm = moments_tab.iloc[:, 1].values.astype(float)
    target_hdg = moments_tab.iloc[:, 2].values.astype(float)

    # ---- Load calibrated parameters ----
    Nsim_final = 10000
    Tsim = len(m)

    print('\n================ LOADING FDM PARAMETERS ================')
    results_fdm = load_smm_params('params_fdm.txt')

    print('\n================ LOADING HEDGING PARAMETERS ================')
    results_hdg = load_smm_params('params_hedging.txt')

    # ---- Simulate from both models ----
    print('\n================ SIMULATING ================')

    rng_fdm = np.random.default_rng(99)
    r_sim_fdm, h_sim_fdm = simulate_model(results_fdm, m, S, V_jump, Tsim, Nsim_final, rng_fdm)

    rng_hdg = np.random.default_rng(99)
    r_sim_hdg, h_sim_hdg = simulate_model(results_hdg, m, S, V_jump, Tsim, Nsim_final, rng_hdg)

    print(f'Simulated {Nsim_final} paths x {Tsim} periods for each portfolio.')

    # ---- Summary statistics ----
    m_dm = m - np.mean(m)
    var_m_scalar = m_dm @ m_dm
    idx_down = m < -0.05
    m_down_dm = m[idx_down] - np.mean(m[idx_down])
    var_m_down = m_down_dm @ m_down_dm

    avg_fdm = compute_moments(r_sim_fdm, m_dm, var_m_scalar, idx_down, m_down_dm, var_m_down)
    avg_hdg = compute_moments(r_sim_hdg, m_dm, var_m_scalar, idx_down, m_down_dm, var_m_down)

    labels = ['Mean (monthly)', 'Mean (annualized)', 'Variance (monthly)',
              'Std Dev (monthly)', 'Std Dev (annualized)', 'Skewness',
              'Excess Kurtosis', 'Jensen Beta', 'Jensen Beta (m<-5%)']

    print('\n================================================================')
    print('          SIMULATED MOMENTS (10k paths)')
    print('================================================================')
    print(f'{"":25s} | {"FDM":>22s} | {"Hedging":>22s}')
    print(f'{"Moment":25s} | {"Target":>10s} {"Sim":>10s} | {"Target":>10s} {"Sim":>10s}')
    print(f'{"-"*25}-|-{"-"*10}-{"-"*10}-|-{"-"*10}-{"-"*10}')
    for i in range(9):
        print(f'{labels[i]:25s} | {target_fdm[i]:10.4f} {avg_fdm[i]:10.4f} '
              f'| {target_hdg[i]:10.4f} {avg_hdg[i]:10.4f}')
    print('================================================================')

    # ---- Parameter comparison ----
    p_fdm = results_fdm['param_hat']
    p_hdg = results_hdg['param_hat']

    param_rows = [
        ('alpha (monthly)',    'alpha'),
        ('beta',               'beta'),
        ('omega',              'omega'),
        ('phi',                'phi'),
        ('sigmah',             'sigmah'),
        ('delta_v (eq vol)',   'delta_v'),
        ('nu (df)',            'nu'),
        ('lambda (skewness)',  'lambda'),
        ('rho (leverage)',     'rho'),
        ('mu_j (jump mean)',   'mu_j'),
        ('psi_0 (jump int.)',  'psi0'),
    ]

    print('\n================================================================')
    print('          CALIBRATED PARAMETERS')
    print('================================================================')
    print(f'{"Parameter":25s} | {"FDM":>12s} | {"Hedging":>12s}')
    print(f'{"-"*25}-|-{"-"*12}-|-{"-"*12}')
    for label, key in param_rows:
        print(f'{label:25s} | {p_fdm[key]:+12.6f} | {p_hdg[key]:+12.6f}')
    for iv in range(len(p_fdm['psi'])):
        print(f'{f"psi_{iv+1}":25s} | {p_fdm["psi"][iv]:+12.6f} | {p_hdg["psi"][iv]:+12.6f}')
    print('================================================================\n')

    print(f'Arrays r_sim_fdm, r_sim_hdg, h_sim_fdm, h_sim_hdg computed.')
    print(f'Each is {Tsim} x {Nsim_final} (time periods x simulated paths).\n')

    return r_sim_fdm, r_sim_hdg, h_sim_fdm, h_sim_hdg, results_fdm, results_hdg


if __name__ == '__main__':
    main()
