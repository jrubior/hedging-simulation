%% run_hedging_skewt_jumpv_smm.m
%  Calibrate SV-X (Skewed-t + Leverage + Vol-Driven Jumps) for Hedging
%  returns using Simulated Method of Moments (SMM).
%
%  Model:
%    r_t = alpha + beta*m_t + gamma'*s_t + exp(0.5*h_t)*eps_t
%          + J_t*Z_t - p_{j,t}*mu_j                  [jump-compensated]
%          (E[J_t*Z_t | F_{t-1}] = p_{j,t}*mu_j)
%
%  Mean equation estimated via OLS (fixed).
%  Distributional parameters calibrated to match 7 target moments.

clear; clc;

%% ---- Load data ----
file = 'hedging_states_returns.xlsx';

% Portfolio returns
ret  = readtable(file, 'Sheet', 'Portfolio_returns');
r    = ret{:,3};          % Column C: Hedging

% States
st   = readtable(file, 'Sheet', 'States');
m    = st{:,3};           % Column C: STOCK_EXCESS_RETURN
S    = [st{:,2}, st{:,4:16}];  % Columns B,D-P: 14 state variables

% Volatilities (4 additional state variables)
dates   = ret{:,1};
vol_tab = readtable('volatilities.xlsx');
vol_d   = vol_tab{:,1};
[~, i_ret, i_vol] = intersect(dates, vol_d);
V = NaN(length(dates), 4);
V(i_ret,:) = vol_tab{i_vol, 2:5};
S = [S, V];                   % 14 + 4 = 18 state variables

% Keep only data up to 12/31/2025
idx   = dates <= datetime(2025,12,31);
r = r(idx);  m = m(idx);  S = S(idx,:);
V_jump = S(:, end-3);         % equity vol only for jump intensity

%% ---- Target moments ----
moments_tab = readtable(file, 'Sheet', 'Moments');
target = moments_tab{:, 3};   % Column C: Hedging moments
% target: (1) Mean monthly, (2) Mean ann, (3) Var monthly,
%         (4) Std monthly, (5) Std ann, (6) Skewness, (7) Excess Kurtosis

%% ---- Settings ----
Nsim    = 5000;       % Simulations per SMM evaluation
Seed    = 42;         % RNG seed
MaxIter = 3000;       % Max fminsearch iterations

% Moment weights: [mean_m, mean_ann, var_m, std_m, std_ann, skew, exkurt, beta, beta_down]
% Higher weight on skewness & kurtosis; skip redundant moments
Weights = [1 0 1 1 0 3 3 2 2];

%% ---- Calibrate ----
results = calibrate_smm_skewt_jumpv(r, m, S, V_jump, target, ...
    'Nsim',       Nsim, ...
    'Seed',       Seed, ...
    'MaxIter',    MaxIter, ...
    'Weights',    Weights);

%% ---- Final simulation with more paths for robust comparison ----
p     = results.param_hat;
Sz    = (S - results.S_mean) ./ results.S_std;
Vz    = (V_jump - results.V_mean) ./ results.V_std;
K     = size(Sz, 2);
Kv    = size(Vz, 2);
Tsim  = length(r);
Nsim_final = 10000;

% Conditional mean (OLS, fixed)
mu_cond = p.alpha + p.beta * m + (K > 0) * (Sz * p.gamma);

% Time-varying jump probability
pj_t = 1 ./ (1 + exp(-(p.psi0 + (Kv > 0) * (Vz * p.psi))));

% Hansen's skewed-t constants
logc = gammaln((p.nu+1)/2) - gammaln(p.nu/2) - 0.5*log(pi*(p.nu-2));
c_skt = exp(logc);
a_skt = 4*p.lambda*c_skt*(p.nu-2)/(p.nu-1);
b_skt = sqrt(max(1 + 3*p.lambda^2 - a_skt^2, 1e-10));

% Sample from Hansen's skewed-t
rng(99);
Z_base = randn(Tsim, Nsim_final);
chi2_draws = reshape(2 * gamrnd(p.nu/2 * ones(Tsim*Nsim_final,1), ones(Tsim*Nsim_final,1)), Tsim, Nsim_final);
V_t = Z_base ./ sqrt(chi2_draws / p.nu);
V_s = V_t * sqrt((p.nu - 2) / p.nu);
U_bern = (rand(Tsim, Nsim_final) < (1 + p.lambda)/2);
W = abs(V_s) .* ((1+p.lambda)*U_bern - (1-p.lambda)*(1-U_bern));
eps_sim = (W - a_skt) / b_skt;

% Simulate h paths with leverage
xi    = randn(Tsim, Nsim_final);
h_sim = zeros(Tsim, Nsim_final);
sqrt_1mrho2 = sqrt(max(1 - p.rho^2, 0));

h_sim(1,:) = p.omega + p.delta_v * Vz(1,:) + p.sigmah / sqrt(max(1 - p.phi^2, 1e-6)) * xi(1,:);
for t = 2:Tsim
    h_sim(t,:) = p.omega + p.phi * (h_sim(t-1,:) - p.omega) + ...
                 p.delta_v * Vz(t,:) + ...
                 p.sigmah * (p.rho * eps_sim(t-1,:) + sqrt_1mrho2 * xi(t,:));
end

% Jump components
J_sim = (rand(Tsim, Nsim_final) < repmat(pj_t, 1, Nsim_final));
Z_sim = (-p.mu_j * log(rand(Tsim, Nsim_final))) .* J_sim;

% Returns (jump-compensated)
r_sim = repmat(mu_cond, 1, Nsim_final) + exp(0.5 * h_sim) .* eps_sim ...
        + Z_sim - repmat(pj_t * p.mu_j, 1, Nsim_final);

% Precompute for Jensen's beta
m_dm = m - mean(m);
var_m_scalar = m_dm' * m_dm;
idx_down = (m < -0.05);
m_down_dm = m(idx_down) - mean(m(idx_down));
var_m_down = m_down_dm' * m_down_dm;

% Compute moments
beta_final     = (m_dm' * r_sim) / var_m_scalar;
beta_down_final = (m_down_dm' * r_sim(idx_down,:)) / var_m_down;

sim_moments = [ mean(r_sim, 1);
                12 * mean(r_sim, 1);
                var(r_sim, 0, 1);
                std(r_sim, 0, 1);
                sqrt(12) * std(r_sim, 0, 1);
                skewness(r_sim, 0, 1);
                kurtosis(r_sim, 0, 1) - 3;
                beta_final;
                beta_down_final ];
avg = mean(sim_moments, 2);

labels = {'Mean (monthly)', 'Mean (annualized)', 'Variance (monthly)', ...
          'Std Dev (monthly)', 'Std Dev (annualized)', 'Skewness', ...
          'Excess Kurtosis', 'Jensen Beta', 'Jensen Beta (m<-5%)'};

fprintf('\n= FINAL MOMENT COMPARISON (Hedging, SMM Skewt+Jump, 10k paths) =\n');
fprintf('%-25s %12s %12s %10s\n', 'Moment', 'Target', 'Simulated', '% Diff');
fprintf('%-25s %12s %12s %10s\n', '-------------------------', '------------', ...
        '------------', '----------');
for i = 1:9
    if abs(target(i)) > 1e-12
        pct = (avg(i) - target(i)) / abs(target(i)) * 100;
    else
        pct = NaN;
    end
    fprintf('%-25s %12.6f %12.6f %+9.2f%%\n', labels{i}, target(i), avg(i), pct);
end
fprintf('================================================================\n\n');
