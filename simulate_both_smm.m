%% simulate_both_smm.m
%  Calibrate and simulate from both FDM and Hedging SV-X models.
%
%  Model:
%    r_t = alpha + beta*m_t + gamma'*s_t + exp(0.5*h_t)*eps_t
%          + J_t*Z_t - p_{j,t}*mu_j                  [jump-compensated]
%    eps_t ~ Hansen_skewt(nu, lambda)
%    xi_t  ~ N(0,1), independent of eps_t
%    h_t = omega + phi*(h_{t-1}-omega) + delta_v*v_t
%          + sigma_h*(rho*eps_{t-1} + sqrt(1-rho^2)*xi_t)
%    p_{j,t} = sigmoid(psi_0 + psi'*v_t)
%    Z_t ~ Exp(mu_j)
%
%  Outputs saved to workspace:
%    r_sim_fdm  - Tsim x Nsim_final simulated FDM returns
%    r_sim_hdg  - Tsim x Nsim_final simulated Hedging returns
%    results_fdm, results_hdg - calibration results

clear; clc;

%% ======== Load data (shared) ========
file = 'hedging_states_returns.xlsx';

ret  = readtable(file, 'Sheet', 'Portfolio_returns');
r_fdm = ret{:,2};        % Column B: FDM
r_hdg = ret{:,3};        % Column C: Hedging

st   = readtable(file, 'Sheet', 'States');
m    = st{:,3};           % Column C: STOCK_EXCESS_RETURN
S    = [st{:,2}, st{:,4:16}];  % Columns B,D-P: 14 state variables

dates   = ret{:,1};
vol_tab = readtable('volatilities.xlsx');
vol_d   = vol_tab{:,1};
[~, i_ret, i_vol] = intersect(dates, vol_d);
V = NaN(length(dates), 4);
V(i_ret,:) = vol_tab{i_vol, 2:5};
S = [S, V];                   % 14 + 4 = 18 state variables

idx   = dates <= datetime(2025,12,31);
r_fdm = r_fdm(idx);  r_hdg = r_hdg(idx);
m = m(idx);  S = S(idx,:);
V_jump = S(:, end-3);

moments_tab = readtable(file, 'Sheet', 'Moments');
target_fdm = moments_tab{:, 2};
target_hdg = moments_tab{:, 3};

%% ======== Load calibrated parameters from files ========
Nsim_final = 10000;
Tsim       = length(m);

fprintf('\n================ LOADING FDM PARAMETERS ================\n');
results_fdm = load_smm_params('params_fdm.txt');

fprintf('\n================ LOADING HEDGING PARAMETERS ================\n');
results_hdg = load_smm_params('params_hedging.txt');

%% ======== Simulate from both models ========
fprintf('\n================ SIMULATING ================\n');

rng(99);
[r_sim_fdm, h_sim_fdm] = simulate_model(results_fdm, m, S, V_jump, Tsim, Nsim_final);

rng(99);
[r_sim_hdg, h_sim_hdg] = simulate_model(results_hdg, m, S, V_jump, Tsim, Nsim_final);

fprintf('Simulated %d paths x %d periods for each portfolio.\n', Nsim_final, Tsim);

%% ======== Summary statistics ========
% Jensen's beta precomputation
m_dm = m - mean(m);
var_m_scalar = m_dm' * m_dm;
idx_down = (m < -0.05);
m_down_dm = m(idx_down) - mean(m(idx_down));
var_m_down = m_down_dm' * m_down_dm;

avg_fdm = compute_moments(r_sim_fdm, m_dm, var_m_scalar, idx_down, m_down_dm, var_m_down);
avg_hdg = compute_moments(r_sim_hdg, m_dm, var_m_scalar, idx_down, m_down_dm, var_m_down);

labels = {'Mean (monthly)', 'Mean (annualized)', 'Variance (monthly)', ...
          'Std Dev (monthly)', 'Std Dev (annualized)', 'Skewness', ...
          'Excess Kurtosis', 'Jensen Beta', 'Jensen Beta (m<-5%)'};

fprintf('\n================================================================\n');
fprintf('          SIMULATED MOMENTS (10k paths)\n');
fprintf('================================================================\n');
fprintf('%-25s | %22s | %22s\n', '', 'FDM', 'Hedging');
fprintf('%-25s | %10s %10s | %10s %10s\n', ...
    'Moment', 'Target', 'Sim', 'Target', 'Sim');
fprintf('%-25s-|-%10s-%10s-|-%10s-%10s\n', ...
    repmat('-',1,25), repmat('-',1,10), repmat('-',1,10), ...
    repmat('-',1,10), repmat('-',1,10));
for i = 1:9
    fprintf('%-25s | %10.4f %10.4f | %10.4f %10.4f\n', ...
        labels{i}, target_fdm(i), avg_fdm(i), target_hdg(i), avg_hdg(i));
end
fprintf('================================================================\n');

%% ---- Parameter comparison ----
p_fdm = results_fdm.param_hat;
p_hdg = results_hdg.param_hat;

fprintf('\n================================================================\n');
fprintf('          CALIBRATED PARAMETERS\n');
fprintf('================================================================\n');
fprintf('%-25s | %12s | %12s\n', 'Parameter', 'FDM', 'Hedging');
fprintf('%-25s-|-%12s-|-%12s\n', ...
    repmat('-',1,25), repmat('-',1,12), repmat('-',1,12));
fprintf('%-25s | %+12.6f | %+12.6f\n', 'alpha (monthly)',   p_fdm.alpha,   p_hdg.alpha);
fprintf('%-25s | %+12.6f | %+12.6f\n', 'beta',              p_fdm.beta,    p_hdg.beta);
fprintf('%-25s | %+12.6f | %+12.6f\n', 'omega',             p_fdm.omega,   p_hdg.omega);
fprintf('%-25s | %+12.6f | %+12.6f\n', 'phi',               p_fdm.phi,     p_hdg.phi);
fprintf('%-25s | %+12.6f | %+12.6f\n', 'sigmah',            p_fdm.sigmah,  p_hdg.sigmah);
fprintf('%-25s | %+12.6f | %+12.6f\n', 'delta_v (eq vol)',  p_fdm.delta_v, p_hdg.delta_v);
fprintf('%-25s | %+12.6f | %+12.6f\n', 'nu (df)',           p_fdm.nu,      p_hdg.nu);
fprintf('%-25s | %+12.6f | %+12.6f\n', 'lambda (skewness)', p_fdm.lambda,  p_hdg.lambda);
fprintf('%-25s | %+12.6f | %+12.6f\n', 'rho (leverage)',    p_fdm.rho,     p_hdg.rho);
fprintf('%-25s | %+12.6f | %+12.6f\n', 'mu_j (jump mean)',  p_fdm.mu_j,   p_hdg.mu_j);
fprintf('%-25s | %+12.6f | %+12.6f\n', 'psi_0 (jump int.)', p_fdm.psi0,   p_hdg.psi0);
for iv = 1:length(p_fdm.psi)
    fprintf('%-25s | %+12.6f | %+12.6f\n', ...
        sprintf('psi_%d', iv), p_fdm.psi(iv), p_hdg.psi(iv));
end
fprintf('================================================================\n\n');

fprintf('Variables r_sim_fdm, r_sim_hdg, h_sim_fdm, h_sim_hdg saved to workspace.\n');
fprintf('Each is %d x %d (time periods x simulated paths).\n\n', Tsim, Nsim_final);


%% ======== Local functions ========

function [r_sim, h_sim] = simulate_model(results, m, S, V_jump, Tsim, Nsim)
%SIMULATE_MODEL  Simulate return paths from a calibrated SV-X model.

    p  = results.param_hat;
    Sz = (S - results.S_mean) ./ results.S_std;
    Vz = (V_jump - results.V_mean) ./ results.V_std;
    K  = size(Sz, 2);
    Kv = size(Vz, 2);

    % Conditional mean (OLS, fixed)
    mu_cond = p.alpha + p.beta * m + (K > 0) * (Sz * p.gamma);

    % Time-varying jump probability
    pj_t = 1 ./ (1 + exp(-(p.psi0 + (Kv > 0) * (Vz * p.psi))));

    % Hansen's skewed-t constants
    logc  = gammaln((p.nu+1)/2) - gammaln(p.nu/2) - 0.5*log(pi*(p.nu-2));
    c_skt = exp(logc);
    a_skt = 4*p.lambda*c_skt*(p.nu-2)/(p.nu-1);
    b_skt = sqrt(max(1 + 3*p.lambda^2 - a_skt^2, 1e-10));

    % Sample from Hansen's skewed-t (continuous nu via gamma draw)
    Z_base     = randn(Tsim, Nsim);
    chi2_draws = reshape(2 * gamrnd(p.nu/2 * ones(Tsim*Nsim,1), ...
                 ones(Tsim*Nsim,1)), Tsim, Nsim);
    V_t    = Z_base ./ sqrt(chi2_draws / p.nu);
    V_s    = V_t * sqrt((p.nu - 2) / p.nu);
    U_bern = (rand(Tsim, Nsim) < (1 + p.lambda)/2);
    W       = abs(V_s) .* ((1+p.lambda)*U_bern - (1-p.lambda)*(1-U_bern));
    eps_sim = (W - a_skt) / b_skt;

    % Simulate h paths with leverage (stationary initialization)
    xi    = randn(Tsim, Nsim);
    h_sim = zeros(Tsim, Nsim);
    sqrt_1mrho2 = sqrt(max(1 - p.rho^2, 0));

    h_sim(1,:) = p.omega + p.delta_v * Vz(1,:) + ...
                 p.sigmah / sqrt(max(1 - p.phi^2, 1e-6)) * xi(1,:);
    for t = 2:Tsim
        h_sim(t,:) = p.omega + p.phi * (h_sim(t-1,:) - p.omega) + ...
                     p.delta_v * Vz(t,:) + ...
                     p.sigmah * (p.rho * eps_sim(t-1,:) + sqrt_1mrho2 * xi(t,:));
    end

    % Jump components
    J_sim = (rand(Tsim, Nsim) < repmat(pj_t, 1, Nsim));
    Z_sim = (-p.mu_j * log(rand(Tsim, Nsim))) .* J_sim;

    % Returns (jump-compensated)
    r_sim = repmat(mu_cond, 1, Nsim) + exp(0.5 * h_sim) .* eps_sim ...
            + Z_sim - repmat(pj_t * p.mu_j, 1, Nsim);
end

function avg = compute_moments(r_sim, m_dm, var_m_scalar, idx_down, m_down_dm, var_m_down)
%COMPUTE_MOMENTS  Compute 9 moments from simulated returns.

    beta_sim      = (m_dm' * r_sim) / var_m_scalar;
    beta_down_sim = (m_down_dm' * r_sim(idx_down,:)) / var_m_down;

    sim_moments = [ mean(r_sim, 1);
                    12 * mean(r_sim, 1);
                    var(r_sim, 0, 1);
                    std(r_sim, 0, 1);
                    sqrt(12) * std(r_sim, 0, 1);
                    skewness(r_sim, 0, 1);
                    kurtosis(r_sim, 0, 1) - 3;
                    beta_sim;
                    beta_down_sim ];
    avg = mean(sim_moments, 2);
end
