%% run_fdm_skewt.m
%  Estimate SV-X (Skewed-t + Leverage) model for FDM portfolio returns.
%
%  Model:
%    r_t = alpha + beta*m_t + gamma'*s_t + exp(0.5*h_t)*eps_t
%    eps_t ~ Hansen_skewt(nu, lambda)   (mean 0, variance 1)
%    h_t = omega + phi*(h_{t-1}-omega) + delta'*s_t
%           + sigma_h*(rho*eps_{t-1} + sqrt(1-rho^2)*xi_t)
%
%  r_t  = Column B of Portfolio_returns  (FDM)
%  m_t  = Column C of States             (STOCK_EXCESS_RETURN)
%  s_t  = Columns B,D-P of States        (14 state variables)

clear; clc;

%% ---- Load data ----
file = 'hedging_states_returns.xlsx';

% Portfolio returns
ret  = readtable(file, 'Sheet', 'Portfolio_returns');
r    = ret{:,2};          % Column B: FDM

% States
st   = readtable(file, 'Sheet', 'States');
m    = st{:,3};           % Column C: STOCK_EXCESS_RETURN
S    = [st{:,2}, st{:,4:16}];  % Columns B,D-P: 14 state variables

% Keep only data up to 12/31/2025
dates = ret{:,1};
idx   = dates <= datetime(2025,12,31);
r = r(idx);  m = m(idx);  S = S(idx,:);

%% ---- Settings ----
NParticles      = 5000;     % Number of particles in the filter
Seed            = 123;      % RNG seed for reproducibility
ComputeStdErr   = true;     % Compute standard errors via numerical Hessian
HessStep        = 1e-4;     % Step size for numerical Hessian
Display         = 'iter';   % fminunc display level ('iter','final','off')
MaxIter         = 300;      % Max optimization iterations
ResampleThresh  = 0.5;      % ESS resampling threshold (fraction of NParticles)

%% ---- Estimate ----
results = fit_svx_skewt_pf_full(r, m, S, ...
    'NParticles',     NParticles, ...
    'Seed',           Seed, ...
    'ComputeStdErr',  ComputeStdErr, ...
    'HessStep',       HessStep, ...
    'Display',        Display, ...
    'MaxIter',        MaxIter, ...
    'ResampleThresh', ResampleThresh);

%% ---- Simulate from model & compare with target moments ----
moments_tab = readtable(file, 'Sheet', 'Moments');
target = moments_tab{:, 2};   % Column B: FDM moments
% target: (1) Mean monthly, (2) Mean ann, (3) Var monthly,
%         (4) Std monthly, (5) Std ann, (6) Skewness, (7) Excess Kurtosis

Nsim  = 10000;
Tsim  = length(r);
p     = results.param_hat;
Sz    = (S - results.S_mean) ./ results.S_std;
K     = size(Sz, 2);

% Conditional mean (deterministic given observed m_t, s_t)
mu_cond = p.alpha + p.beta * m + (K > 0) * (Sz * p.gamma);

% Hansen's skewed-t constants for sampling
logc = gammaln((p.nu+1)/2) - gammaln(p.nu/2) - 0.5*log(pi*(p.nu-2));
c_skt = exp(logc);
a_skt = 4*p.lambda*c_skt*(p.nu-2)/(p.nu-1);
b_skt = sqrt(1 + 3*p.lambda^2 - a_skt^2);

% Sample from Hansen's skewed-t: mean 0, variance 1
rng(42);
V = trnd(p.nu, Tsim, Nsim);                       % standard t_nu
V_s = V * sqrt((p.nu - 2) / p.nu);                % standardized (variance-1 t)
U = (rand(Tsim, Nsim) < (1 + p.lambda)/2);        % Bernoulli
W = abs(V_s) .* ((1+p.lambda)*U - (1-p.lambda)*(1-U));
eps_sim = (W - a_skt) / b_skt;                    % skewed-t(nu,lambda), E=0, V=1

% Simulate log-volatility paths with leverage: Tsim x Nsim
xi    = randn(Tsim, Nsim);
h_sim = zeros(Tsim, Nsim);
sqrt_1mrho2 = sqrt(max(1 - p.rho^2, 0));

h_sim(1,:) = p.omega + (K > 0) * (Sz(1,:) * p.delta) + xi(1,:);
for t = 2:Tsim
    h_sim(t,:) = p.omega + p.phi * (h_sim(t-1,:) - p.omega) + ...
                 (K > 0) * (Sz(t,:) * p.delta) + ...
                 p.sigmah * (p.rho * eps_sim(t-1,:) + sqrt_1mrho2 * xi(t,:));
end

% Simulated returns: Tsim x Nsim
r_sim = repmat(mu_cond, 1, Nsim) + exp(0.5 * h_sim) .* eps_sim;

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

fprintf('\n==== MOMENT COMPARISON: SIMULATED vs TARGET (FDM, Skewed-t) ====\n');
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
fprintf('=================================================================\n\n');
