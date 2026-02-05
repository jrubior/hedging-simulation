%% run_hedging_jumpv.m
%  Estimate SV-X (Normal + Vol-Driven Jumps) model for Hedging returns.
%
%  Model:
%    r_t = alpha + beta*m_t + gamma'*s_t + exp(0.5*h_t)*eps_t + J_t*Z_t
%    eps_t ~ N(0,1)
%    J_t   ~ Bernoulli(p_{j,t})     p_{j,t} = sigmoid(psi_0 + psi'*v_t)
%    Z_t   ~ Exp(1/mu_j)
%    h_t = omega + phi*(h_{t-1}-omega) + delta'*s_t + sigma_h*xi_t
%
%  s_t  = 18 state variables (14 original + 4 volatilities)
%  v_t  = equity volatility (drives jump intensity only)

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

%% ---- Settings ----
NParticles      = 5000;     % Number of particles in the filter
Seed            = 123;      % RNG seed for reproducibility
ComputeStdErr   = true;     % Compute standard errors via numerical Hessian
HessStep        = 1e-4;     % Step size for numerical Hessian
Display         = 'iter';   % fminunc display level ('iter','final','off')
MaxIter         = 400;      % Max optimization iterations
ResampleThresh  = 0.5;      % ESS resampling threshold (fraction of NParticles)

%% ---- Estimate ----
results = fit_svx_jumpv_pf_full(r, m, S, V_jump, ...
    'NParticles',     NParticles, ...
    'Seed',           Seed, ...
    'ComputeStdErr',  ComputeStdErr, ...
    'HessStep',       HessStep, ...
    'Display',        Display, ...
    'MaxIter',        MaxIter, ...
    'ResampleThresh', ResampleThresh);

%% ---- Simulate from model & compare with target moments ----
moments_tab = readtable(file, 'Sheet', 'Moments');
target = moments_tab{:, 3};   % Column C: Hedging moments

Nsim  = 10000;
Tsim  = length(r);
p     = results.param_hat;
Sz    = (S - results.S_mean) ./ results.S_std;
Vz    = (V_jump - results.V_mean) ./ results.V_std;
K     = size(Sz, 2);

% Conditional mean (deterministic given observed m_t, s_t)
mu_cond = p.alpha + p.beta * m + (K > 0) * (Sz * p.gamma);

% Time-varying jump probability from vol states: Tsim x 1
Kv   = size(Vz, 2);
pj_t = 1 ./ (1 + exp(-(p.psi0 + (Kv > 0) * (Vz * p.psi))));

% Simulate log-volatility paths: Tsim x Nsim
rng(42);
xi    = randn(Tsim, Nsim);
h_sim = zeros(Tsim, Nsim);
h_sim(1,:) = p.omega + (K > 0) * (Sz(1,:) * p.delta) + xi(1,:);
for t = 2:Tsim
    h_sim(t,:) = p.omega + p.phi * (h_sim(t-1,:) - p.omega) + ...
                 (K > 0) * (Sz(t,:) * p.delta) + p.sigmah * xi(t,:);
end

% Draw Normal innovations
eps_sim = randn(Tsim, Nsim);

% Draw vol-driven jump components
J_sim = (rand(Tsim, Nsim) < repmat(pj_t, 1, Nsim));
Z_sim = exprnd(p.mu_j, Tsim, Nsim) .* J_sim;

% Simulated returns: Tsim x Nsim
r_sim = repmat(mu_cond, 1, Nsim) + exp(0.5 * h_sim) .* eps_sim + Z_sim;

% Compute moments per simulated path (along dim 1), then average
sim_moments = [ mean(r_sim, 1);              % mean monthly
                12 * mean(r_sim, 1);         % mean annualized
                var(r_sim, 0, 1);            % variance monthly
                std(r_sim, 0, 1);            % std monthly
                sqrt(12) * std(r_sim, 0, 1); % std annualized
                skewness(r_sim, 0, 1);       % skewness
                kurtosis(r_sim, 0, 1) - 3 ]; % excess kurtosis
avg = mean(sim_moments, 2);

labels = {'Mean (monthly)', 'Mean (annualized)', 'Variance (monthly)', ...
          'Std Dev (monthly)', 'Std Dev (annualized)', 'Skewness', ...
          'Excess Kurtosis'};

fprintf('\n= MOMENT COMPARISON: SIMULATED vs TARGET (Hedging, Vol Jumps) =\n');
fprintf('%-25s %12s %12s %10s\n', 'Moment', 'Target', 'Simulated', '% Diff');
fprintf('%-25s %12s %12s %10s\n', '-------------------------', '------------', ...
        '------------', '----------');
for i = 1:7
    if abs(target(i)) > 1e-12
        pct = (avg(i) - target(i)) / abs(target(i)) * 100;
    else
        pct = NaN;
    end
    fprintf('%-25s %12.6f %12.6f %+9.2f%%\n', labels{i}, target(i), avg(i), pct);
end
fprintf('==================================================================\n\n');
