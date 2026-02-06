
function results = calibrate_smm_skewt_jumpv(r, m, S, V, target_moments, varargin)
%CALIBRATE_SMM_SKEWT_JUMPV  Moment-matching calibration for SV-X model
% with Skewed-t + Leverage + Vol-Driven Jumps.
%
% Instead of MLE, calibrates distributional parameters to minimize the
% distance between simulated and target unconditional moments.
%
% Model:
%   r_t = alpha + beta*m_t + gamma'*s_t + exp(0.5*h_t)*eps_t
%         + J_t*Z_t - p_{j,t}*mu_j                  [jump-compensated]
%         (E[J_t*Z_t | F_{t-1}] = p_{j,t}*mu_j)
%   eps_t ~ Hansen_skewt(nu, lambda)
%   xi_t  ~ N(0,1), independent of eps_t
%   h_t = omega + phi*(h_{t-1}-omega) + delta_v*v_t
%         + sigma_h*(rho*eps_{t-1} + sqrt(1-rho^2)*xi_t)
%   p_{j,t} = sigmoid(psi_0 + psi'*v_t)
%   Z_t ~ Exp(mu_j)
%
% Stage 1: Mean equation (alpha, beta, gamma) via OLS — held fixed.
% Stage 2: Distributional params via Simulated Method of Moments:
%   [omega, phi, sigmah, delta_v, nu, lambda, rho, psi0, psi(Kv), mu_j]
%
% Arguments:
%   r  - T x 1 returns
%   m  - T x 1 market returns
%   S  - T x K state variables
%   V  - T x Kv volatility states (jump intensity + vol equation via delta_v)
%   target_moments - 9 x 1: [mean_m; mean_ann; var_m; std_m; std_ann; skew; exkurt; beta; beta_down]
%
% Options:
%   'Nsim'        (default 5000) simulations per evaluation
%   'Seed'        (default 42)   RNG seed for simulation
%   'MaxIter'     (default 3000) max fminsearch iterations
%   'Display'     (default 'iter')
%   'Weights'     (default [1 0 1 1 0 3 3]) moment weights
%   'InitParams'  (default struct()) override initial distributional params
% ------------------------------------------------------------

    if nargin < 3, S = []; end
    if nargin < 4, V = []; end
    r = r(:); m = m(:);
    target_moments = target_moments(:);
    T = length(r);
    K  = size(S,2);
    Kv = size(V,2);

    % -------- Parse options --------
    pOpt = inputParser;
    pOpt.addParameter('Nsim', 5000);
    pOpt.addParameter('Seed', 42);
    pOpt.addParameter('MaxIter', 3000);
    pOpt.addParameter('Display', 'iter');
    pOpt.addParameter('Weights', [1 0 1 1 0 3 3 2 2]);
    pOpt.addParameter('InitParams', struct());
    pOpt.parse(varargin{:});
    opt = pOpt.Results;

    Nsim = opt.Nsim;
    seed = opt.Seed;
    weights = opt.Weights(:);
    Tsim = T;

    % -------- Precompute for Jensen's beta moments --------
    m_dm = m - mean(m);
    var_m_scalar = m_dm' * m_dm;            % T * Var(m)
    idx_down = (m < -0.05);
    m_down_dm = m(idx_down) - mean(m(idx_down));
    var_m_down = m_down_dm' * m_down_dm;    % T_down * Var(m_down)

    % -------- Standardize states --------
    S_mean = zeros(1,K); S_std = ones(1,K);
    if K > 0
        S_mean = mean(S,1,'omitnan');
        S_std  = std(S,0,1,'omitnan');
        S_std(S_std==0) = 1;
        Sz = (S - S_mean)./S_std;
    else
        Sz = zeros(T,0);
    end

    V_mean = zeros(1,Kv); V_std = ones(1,Kv);
    if Kv > 0
        V_mean = mean(V,1,'omitnan');
        V_std  = std(V,0,1,'omitnan');
        V_std(V_std==0) = 1;
        Vz = (V - V_mean)./V_std;
    else
        Vz = zeros(T,0);
    end

    % -------- Stage 1: OLS for mean equation --------
    X = [ones(T,1) m Sz];
    b_ols = X \ r;
    alpha_hat = b_ols(1);
    beta_hat  = b_ols(2);
    gamma_hat = b_ols(3:end);

    mu_cond = alpha_hat + beta_hat * m + (K>0)*(Sz * gamma_hat);

    % -------- Initial distributional parameters --------
    resid = r - mu_cond;
    v0 = var(resid, 'omitnan');
    if ~(isfinite(v0) && v0>0), v0 = 1e-4; end

    omega0   = log(v0);
    phi0     = 0.85;
    sigmah0  = 0.30;
    deltav0  = 0;
    nu0      = 5;
    lambda0  = 0.3;
    rho0     = -0.1;
    psi00    = log(0.10/0.90);    % logit(0.10)
    psi0_vec = zeros(Kv,1);
    muj0     = 0.08;

    % Override with user-supplied InitParams
    ip = opt.InitParams;
    if isfield(ip,'omega'),   omega0   = ip.omega;   end
    if isfield(ip,'phi'),     phi0     = ip.phi;     end
    if isfield(ip,'sigmah'),  sigmah0  = ip.sigmah;  end
    if isfield(ip,'delta_v'), deltav0  = ip.delta_v; end
    if isfield(ip,'nu'),      nu0      = ip.nu;      end
    if isfield(ip,'lambda'),  lambda0  = ip.lambda;  end
    if isfield(ip,'rho'),     rho0     = ip.rho;     end
    if isfield(ip,'mu_j'),    muj0     = ip.mu_j;    end
    if isfield(ip,'psi0'),    psi00    = ip.psi0;    end

    theta0 = pack_dist(omega0, phi0, sigmah0, deltav0, nu0, lambda0, rho0, ...
                        psi00, psi0_vec, muj0);

    % -------- Normalization for objective --------
    norm_fac = max(abs(target_moments), 0.001);

    % -------- Optimize via fminsearch --------
    fms_opts = optimset('Display', opt.Display, ...
                        'MaxIter', opt.MaxIter, ...
                        'MaxFunEvals', 10000, ...
                        'TolFun', 1e-8, ...
                        'TolX', 1e-6);

    [theta_hat, fval, exitflag, output] = ...
        fminsearch(@smm_obj, theta0, fms_opts);

    % -------- Final simulation at calibrated parameters --------
    pd = unpack_dist(theta_hat, Kv);
    [~, final_moments] = smm_obj(theta_hat);

    % -------- Package results --------
    param_hat = struct();
    param_hat.alpha  = alpha_hat;
    param_hat.beta   = beta_hat;
    param_hat.gamma  = gamma_hat;
    param_hat.omega  = pd.omega;
    param_hat.phi    = pd.phi;
    param_hat.sigmah = pd.sigmah;
    param_hat.delta_v = pd.delta_v;
    param_hat.nu     = pd.nu;
    param_hat.lambda = pd.lambda;
    param_hat.rho    = pd.rho;
    param_hat.psi0   = pd.psi0;
    param_hat.psi    = pd.psi;
    param_hat.mu_j   = pd.mu_j;

    results = struct();
    results.param_hat = param_hat;
    results.theta_dist = theta_hat;
    results.smm_obj   = fval;
    results.exitflag  = exitflag;
    results.output    = output;
    results.S_mean    = S_mean;
    results.S_std     = S_std;
    results.V_mean    = V_mean;
    results.V_std     = V_std;
    results.target_moments = target_moments;
    results.simulated_moments = final_moments;

    % -------- Print report --------
    print_smm_report(results);

    % ====================================================================
    %  Nested objective: simulate and compute moment distance
    % ====================================================================
    function [obj, sim_avg] = smm_obj(theta_dist)
        pd_ = unpack_dist(theta_dist, Kv);

        rng(seed, 'twister');

        % --- Hansen's skewed-t constants ---
        logc_ = gammaln((pd_.nu+1)/2) - gammaln(pd_.nu/2) ...
                - 0.5*log(pi*(pd_.nu-2));
        c_skt_ = exp(logc_);
        a_skt_ = 4*pd_.lambda*c_skt_*(pd_.nu-2)/(pd_.nu-1);
        b_skt_ = sqrt(max(1 + 3*pd_.lambda^2 - a_skt_^2, 1e-10));

        % --- Generate t_nu via gamma draw (continuous nu) ---
        Z_ = randn(Tsim, Nsim);
        chi2_ = reshape(2 * gamrnd(pd_.nu/2 * ones(Tsim*Nsim,1), ones(Tsim*Nsim,1)), Tsim, Nsim);
        t_ = Z_ ./ sqrt(chi2_ / pd_.nu);
        Vs_ = t_ * sqrt((pd_.nu - 2) / pd_.nu);

        % --- Hansen's skewed-t transform ---
        U_ = (rand(Tsim, Nsim) < (1 + pd_.lambda)/2);
        W_ = abs(Vs_) .* ((1+pd_.lambda)*U_ - (1-pd_.lambda)*(1-U_));
        eps_ = (W_ - a_skt_) / b_skt_;

        % --- Jump probabilities ---
        pj_ = 1 ./ (1 + exp(-(pd_.psi0 + (Kv>0)*(Vz*pd_.psi))));

        % --- Simulate h paths with leverage ---
        xi_ = randn(Tsim, Nsim);
        h_ = zeros(Tsim, Nsim);
        sqrt1mr2_ = sqrt(max(1 - pd_.rho^2, 0));

        h_(1,:) = pd_.omega + pd_.delta_v * Vz(1,:) + pd_.sigmah / sqrt(max(1 - pd_.phi^2, 1e-6)) * xi_(1,:);
        for t_ = 2:Tsim
            h_(t_,:) = pd_.omega + pd_.phi * (h_(t_-1,:) - pd_.omega) + ...
                       pd_.delta_v * Vz(t_,:) + ...
                       pd_.sigmah * (pd_.rho * eps_(t_-1,:) + sqrt1mr2_ * xi_(t_,:));
        end

        % --- Jump components ---
        J_ = (rand(Tsim, Nsim) < repmat(pj_, 1, Nsim));
        Z_j = (-pd_.mu_j * log(rand(Tsim, Nsim))) .* J_;

        % --- Returns (jump-compensated) ---
        r_sim_ = repmat(mu_cond, 1, Nsim) + exp(0.5 * h_) .* eps_ ...
                 + Z_j - repmat(pj_ * pd_.mu_j, 1, Nsim);

        % --- Compute 9 moments ---
        beta_sim_     = (m_dm' * r_sim_) / var_m_scalar;               % 1 x Nsim
        beta_down_sim = (m_down_dm' * r_sim_(idx_down,:)) / var_m_down; % 1 x Nsim

        sim_mom = [mean(r_sim_, 1);
                   12 * mean(r_sim_, 1);
                   var(r_sim_, 0, 1);
                   std(r_sim_, 0, 1);
                   sqrt(12) * std(r_sim_, 0, 1);
                   skewness(r_sim_, 0, 1);
                   kurtosis(r_sim_, 0, 1) - 3;
                   beta_sim_;
                   beta_down_sim];
        sim_avg = mean(sim_mom, 2);

        % --- Weighted squared relative deviation ---
        rel_dev = (sim_avg - target_moments) ./ norm_fac;
        obj = sum(weights .* rel_dev.^2);
    end
end

%% ============================================================
%  Print SMM calibration report
%% ============================================================
function print_smm_report(res)
    p = res.param_hat;
    target = res.target_moments;
    sim    = res.simulated_moments;

    fprintf('\n====== SMM CALIBRATION (Skewed-t + Leverage + Vol Jumps) ======\n');
    fprintf('SMM objective: %.8f\n', res.smm_obj);

    fprintf('\n--- Mean equation (OLS, fixed) ---\n');
    fprintf('alpha  (monthly):            %8.6f\n', p.alpha);
    fprintf('beta   :                     %8.6f\n', p.beta);

    fprintf('\n--- Calibrated distributional parameters ---\n');
    fprintf('omega  :                     %8.6f\n', p.omega);
    fprintf('phi    :                     %8.6f\n', p.phi);
    fprintf('sigmah :                     %8.6f\n', p.sigmah);
    fprintf('delta_v (eq vol in h_t):     %+8.6f\n', p.delta_v);
    fprintf('nu     (df):                 %8.6f\n', p.nu);
    fprintf('lambda (skewness):           %8.6f\n', p.lambda);
    fprintf('rho    (leverage):           %8.6f\n', p.rho);
    fprintf('mu_j   (jump mean):          %8.6f\n', p.mu_j);
    fprintf('psi_0  (jump intercept):     %8.6f\n', p.psi0);
    if ~isempty(p.psi)
        for iv = 1:length(p.psi)
            fprintf('psi_%-2d :                     %+8.6f\n', iv, p.psi(iv));
        end
    end

    fprintf('\n--- Moment comparison ---\n');
    labels = {'Mean (monthly)', 'Mean (annualized)', 'Variance (monthly)', ...
              'Std Dev (monthly)', 'Std Dev (annualized)', 'Skewness', ...
              'Excess Kurtosis', 'Jensen Beta', 'Jensen Beta (m<-5%)'};
    fprintf('%-25s %12s %12s %10s\n', 'Moment', 'Target', 'Calibrated', '% Diff');
    fprintf('%-25s %12s %12s %10s\n', '-------------------------', '------------', ...
            '------------', '----------');
    for i = 1:9
        if abs(target(i)) > 1e-12
            pct = (sim(i) - target(i)) / abs(target(i)) * 100;
        else
            pct = NaN;
        end
        fprintf('%-25s %12.6f %12.6f %+9.2f%%\n', labels{i}, target(i), sim(i), pct);
    end
    fprintf('==============================================================\n\n');
end

%% ============================================================
%  Parameter packing/unpacking for distributional params
%  theta_dist = [omega; eta_phi; eta_sh; delta_v; eta_nu; eta_lam; eta_rho;
%                psi0; psi(Kv); eta_muj]
%% ============================================================
function theta = pack_dist(omega, phi, sigmah, delta_v, nu, lambda, rho, psi0, psi, muj)
    eta_phi = atanh(max(min(phi,0.999),-0.999));
    eta_sh  = log(max(sigmah, 1e-6));
    eta_nu  = log(max(nu-2, 1e-6));
    eta_lam = atanh(max(min(lambda,0.999),-0.999));
    eta_rho = atanh(max(min(rho,0.999),-0.999));
    eta_muj = log(max(muj - 0.01, 1e-6));
    theta = [omega; eta_phi; eta_sh; delta_v; eta_nu; eta_lam; eta_rho; ...
             psi0; psi(:); eta_muj];
end

function pd = unpack_dist(theta, Kv)
    idx = 0;
    pd = struct();
    idx = idx+1; pd.omega   = theta(idx);
    idx = idx+1; pd.phi     = tanh(theta(idx));
    idx = idx+1; pd.sigmah  = exp(theta(idx));
    idx = idx+1; pd.delta_v = theta(idx);
    idx = idx+1; pd.nu      = 2 + exp(theta(idx));
    idx = idx+1; pd.lambda  = tanh(theta(idx));
    idx = idx+1; pd.rho     = tanh(theta(idx));
    idx = idx+1; pd.psi0    = theta(idx);
    if Kv > 0
        pd.psi = theta(idx+1:idx+Kv); idx = idx + Kv;
    else
        pd.psi = zeros(0,1);
    end
    idx = idx+1; pd.mu_j    = 0.01 + exp(theta(idx));
end
