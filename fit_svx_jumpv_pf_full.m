
function results = fit_svx_jumpv_pf_full(r, m, S, V, varargin)
%FIT_SVX_JUMPV_PF_FULL  SV-X with Normal + volatility-driven Exponential jumps.
%
% Model:
%   r_t = alpha + beta*m_t + gamma'*s_t + exp(0.5*h_t)*eps_t + J_t*Z_t
%   eps_t ~ N(0,1)
%   J_t   ~ Bernoulli(p_{j,t})
%   Z_t   ~ Exp(1/mu_j)            [jump size, mean mu_j > 0]
%   h_t = omega + phi*(h_{t-1}-omega) + delta'*s_t + sigma_h*xi_t
%   xi_t  ~ N(0,1)
%
% Jump intensity driven by volatility states:
%   p_{j,t} = sigmoid(psi_0 + psi'*v_t)
%
% Arguments:
%   r  - T x 1 portfolio returns
%   m  - T x 1 market excess return
%   S  - T x K state variables (for mean and vol equations)
%   V  - T x Kv volatility states (for jump intensity equation)
%
% Parameters:
%   Mean eq:  alpha, beta, gamma (K)
%   Vol eq:   omega, phi, sigma_h, delta (K)
%   Jump eq:  psi_0, psi (Kv), mu_j
%
% Optional name-value pairs:
%   'NParticles'      (default 5000)
%   'Seed'            (default 123)
%   'ComputeStdErr'   (default true)
%   'HessStep'        (default 1e-4)
%   'Display'         (default 'iter')
%   'MaxIter'         (default 400)
%   'ResampleThresh'  (default 0.5)
%
% ------------------------------------------------------------

    if nargin < 3, S = []; end
    if nargin < 4, V = []; end
    r = r(:); m = m(:);
    T = length(r);
    if length(m) ~= T, error('m must have same length as r'); end
    if ~isempty(S) && size(S,1) ~= T, error('S must have T rows'); end
    if ~isempty(V) && size(V,1) ~= T, error('V must have T rows'); end
    K  = size(S,2);
    Kv = size(V,2);

    % -------- Parse options --------
    pOpt = inputParser;
    pOpt.addParameter('NParticles', 5000);
    pOpt.addParameter('Seed', 123);
    pOpt.addParameter('ComputeStdErr', true);
    pOpt.addParameter('HessStep', 1e-4);
    pOpt.addParameter('Display', 'iter');
    pOpt.addParameter('MaxIter', 400);
    pOpt.addParameter('ResampleThresh', 0.5);
    pOpt.parse(varargin{:});
    opt = pOpt.Results;

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

    % -------- Initial values via OLS --------
    X = [ones(T,1) m Sz];
    b_ols = X \ r;

    alpha0 = b_ols(1);
    beta0  = b_ols(2);
    gamma0 = b_ols(3:end);

    resid = r - X*b_ols;
    v0 = var(resid, 'omitnan');
    if ~(isfinite(v0) && v0>0), v0 = 1e-4; end

    omega0   = log(v0);
    phi0     = 0.90;
    sigmah0  = 0.20;
    delta0   = zeros(K,1);
    psi00    = log(0.05/0.95);   % logit(0.05)
    psi0     = zeros(Kv,1);     % no vol dependence initially
    muj0     = 0.05;            % 5% average positive jump

    theta0 = pack_params(alpha0, beta0, gamma0, omega0, phi0, sigmah0, ...
                          delta0, psi00, psi0, muj0);

    % -------- PF settings --------
    pf.N = opt.NParticles;
    pf.seed = opt.Seed;
    pf.h0_var = 1.0;
    pf.resample_thresh = opt.ResampleThresh;

    % -------- Optimize --------
    opts = optimoptions('fminunc',...
        'Algorithm','quasi-newton',...
        'Display', opt.Display,...
        'MaxIterations', opt.MaxIter,...
        'MaxFunctionEvaluations', 8000,...
        'OptimalityTolerance', 1e-6,...
        'StepTolerance', 1e-8);

    obj = @(th) negloglik_pf(th, r, m, Sz, Vz, pf, K, Kv);
    [theta_hat, fval, exitflag, output] = fminunc(obj, theta0, opts);

    % -------- Decode parameters --------
    param_hat = unpack_params(theta_hat, K, Kv);

    % -------- Run PF at MLE to get filtered summaries --------
    [nll, pf_out] = negloglik_pf(theta_hat, r, m, Sz, Vz, pf, K, Kv);

    % -------- Compute reported statistics --------
    if nll > 1e7
        warning('PF failed at MLE solution; statistics may be unreliable.');
    end
    stats = compute_report_stats(r, m, Sz, param_hat, pf_out);

    % -------- Optional: standard errors via numerical Hessian --------
    se = [];
    vcov = [];
    if opt.ComputeStdErr
        hstep = opt.HessStep;
        H = numhess(@(th) negloglik_pf(th, r, m, Sz, Vz, pf, K, Kv), theta_hat, hstep);
        ridge = 1e-8;
        H2 = (H + H')/2 + ridge*eye(size(H));
        [~,pdflag] = chol(H2);
        if pdflag==0
            vcov = inv(H2);
            se = sqrt(max(diag(vcov),0));
        else
            vcov = [];
            se = [];
            warning('Hessian not PD; standard errors not computed.');
        end
    end

    % -------- Package results --------
    results = struct();
    results.theta_hat = theta_hat;
    results.param_hat = param_hat;
    results.negloglik = nll;
    results.exitflag  = exitflag;
    results.output    = output;
    results.pf_out    = pf_out;
    results.stats     = stats;
    results.vcov_theta = vcov;
    results.se_theta   = se;
    results.S_mean = S_mean;
    results.S_std  = S_std;
    results.V_mean = V_mean;
    results.V_std  = V_std;

    print_report(results);
end

%% ============================================================
%  Compute report statistics
%% ============================================================
function stats = compute_report_stats(r, m, Sz, p, pf_out)
    T = length(r);
    K = size(Sz,2);

    mu_hat = p.alpha + p.beta*m + (K>0)*(Sz*p.gamma);
    h_hat  = pf_out.h_mean;
    sig_hat = exp(0.5*h_hat);
    eps_hat = (r - mu_hat) ./ sig_hat;

    mean_m = mean(r,'omitnan');
    vol_m  = std(r,0,'omitnan');
    ann_ret = 12*mean_m;
    ann_vol = sqrt(12)*vol_m;

    neg = (r < 0) & isfinite(r);
    if any(neg)
        dvol_m = std(r(neg),0,'omitnan');
        ann_dvol = sqrt(12)*dvol_m;
        sortino = (12*mean_m) / (sqrt(12)*dvol_m);
    else
        ann_dvol = NaN;
        sortino  = NaN;
    end

    Xcapm = [ones(T,1) m];
    bcapm = Xcapm \ r;
    alpha_ols_m = bcapm(1);
    beta_ols    = bcapm(2);

    rv = r(isfinite(r));
    ev = eps_hat(isfinite(eps_hat));
    sk = skewness(rv, 0);
    ku = kurtosis(rv, 0);
    sk_eps = skewness(ev, 0);
    ku_eps = kurtosis(ev, 0);

    stats = struct();
    stats.ann_return = ann_ret;
    stats.ann_vol = ann_vol;
    stats.ann_downside_vol = ann_dvol;
    stats.sortino = sortino;
    stats.beta_model = p.beta;
    stats.beta_ols = beta_ols;
    stats.alpha_model_monthly = p.alpha;
    stats.alpha_model_annual = 12*p.alpha;
    stats.alpha_ols_monthly = alpha_ols_m;
    stats.alpha_ols_annual  = 12*alpha_ols_m;
    stats.sample_skewness = sk;
    stats.sample_kurtosis = ku;
    stats.stdresid_skewness = sk_eps;
    stats.stdresid_kurtosis = ku_eps;
    stats.ann_vol_fitted = sqrt(12)*std(sig_hat.*eps_hat,0,'omitnan'); %#ok<NASGU>
end

function print_report(res)
    s = res.stats;
    p = res.param_hat;
    pj = res.pf_out.pj_path;

    fprintf('\n======= SV-X (Normal + Vol-Driven Jump) REPORT =======\n');
    fprintf('Log-likelihood: %.4f\n', -res.negloglik);

    fprintf('\n--- Performance stats (from realized r_t) ---\n');
    fprintf('Annualised return (ex cash): %8.4f\n', s.ann_return);
    fprintf('Annualised volatility:       %8.4f\n', s.ann_vol);
    fprintf('Downside volatility (r<0):   %8.4f\n', s.ann_downside_vol);
    fprintf('Sortino ratio (ex cash):     %8.4f\n', s.sortino);

    fprintf('\n--- Market exposure / alpha ---\n');
    fprintf('Beta (model):                %8.4f\n', s.beta_model);
    fprintf('Beta (OLS r~1+m):            %8.4f\n', s.beta_ols);
    fprintf('Jensen alpha (model, annual):%8.4f\n', s.alpha_model_annual);
    fprintf('Jensen alpha (OLS, annual):  %8.4f\n', s.alpha_ols_annual);

    fprintf('\n--- Higher moments (sample, not %% ) ---\n');
    fprintf('Skewness(r):                 %8.4f\n', s.sample_skewness);
    fprintf('Kurtosis(r):                 %8.4f\n', s.sample_kurtosis);
    fprintf('Skewness(std resid):         %8.4f\n', s.stdresid_skewness);
    fprintf('Kurtosis(std resid):         %8.4f\n', s.stdresid_kurtosis);

    fprintf('\n--- Estimated parameters ---\n');
    fprintf('alpha  (monthly):            %8.6f\n', p.alpha);
    fprintf('beta   :                     %8.6f\n', p.beta);
    fprintf('omega  :                     %8.6f\n', p.omega);
    fprintf('phi    :                     %8.6f\n', p.phi);
    fprintf('sigmah :                     %8.6f\n', p.sigmah);
    fprintf('mu_j   (jump mean):          %8.6f\n', p.mu_j);

    fprintf('\n--- Jump intensity: p_{j,t} = sigmoid(psi_0 + psi''*v_t) ---\n');
    fprintf('psi_0  (intercept):          %8.6f\n', p.psi0);
    if ~isempty(p.psi)
        vol_names = {'equities_vol', 'bonds_vol', 'commod_vol', 'infl_vol'};
        for iv = 1:length(p.psi)
            if iv <= length(vol_names)
                fprintf('psi_%-2d (%s):  %+10.6f\n', iv, vol_names{iv}, p.psi(iv));
            else
                fprintf('psi_%-2d :                    %+10.6f\n', iv, p.psi(iv));
            end
        end
    end
    fprintf('Avg jump prob:               %8.4f\n', mean(pj));
    fprintf('Jump prob range:             [%.4f, %.4f]\n', min(pj), max(pj));

    if ~isempty(p.gamma)
        fprintf('gamma (mean states):         [%s]\n', sprintf('%.6f ', p.gamma));
    end
    if ~isempty(p.delta)
        fprintf('delta (vol  states):         [%s]\n', sprintf('%.6f ', p.delta));
    end

    if ~isempty(res.se_theta)
        fprintf('\n--- Approx. std. errors (theta parametrization) ---\n');
        fprintf('se(theta):                   [%s]\n', sprintf('%.6f ', res.se_theta));
    end
    fprintf('======================================================\n\n');
end

%% ============================================================
%  Negative log-likelihood via bootstrap particle filter
%  (Normal innovations + vol-driven Exponential jumps)
%  Observation density:
%    p(r_t|h_t) = (1-p_{j,t})*Normal + p_{j,t}*EMG
%  where p_{j,t} = sigmoid(psi_0 + psi'*v_t)
%% ============================================================
function [nll, out] = negloglik_pf(theta, r, m, Sz, Vz, pf, K, Kv)
    T = length(r);
    p = unpack_params(theta, K, Kv);

    alpha  = p.alpha;
    beta   = p.beta;
    gamma  = p.gamma;
    omega  = p.omega;
    phi    = p.phi;
    sigmah = p.sigmah;
    delta  = p.delta;
    psi0   = p.psi0;
    psi    = p.psi;
    mu_j   = p.mu_j;

    N = pf.N;
    rng(pf.seed, 'twister');

    mu = alpha + beta*m + (K>0)*(Sz*gamma);
    lam = 1/mu_j;
    log_lam     = log(lam);
    half_log2pi = 0.5*log(2*pi);

    % Precompute time-varying jump probabilities from vol states
    pj_path = 1 ./ (1 + exp(-(psi0 + (Kv>0)*(Vz*psi))));  % T x 1

    % Initialize particles for h_1
    if T >= 1
        mean_h1 = omega + (K>0)*(Sz(1,:)*delta);
        h = mean_h1 + sqrt(pf.h0_var)*randn(N,1);
    else
        h = omega + sqrt(pf.h0_var)*randn(N,1);
    end

    loglik = 0;
    ess_path = zeros(T,1);
    h_mean = zeros(T,1);
    h_var  = zeros(T,1);

    for t = 1:T
        if t > 1
            mean_ht = omega + phi*(h - omega) + (K>0)*(Sz(t,:)*delta);
            h = mean_ht + sigmah*randn(N,1);
        end

        sigma = exp(0.5*h);
        e = (r(t) - mu(t)) ./ sigma;

        % Time-varying jump probability at t
        pjt = pj_path(t);
        log_1mpjt = log(max(1 - pjt, 1e-15));
        log_pjt   = log(max(pjt, 1e-15));

        % --- No-jump component: Normal ---
        log_f0 = -half_log2pi - log(sigma) - 0.5*e.^2;

        % --- Jump component: EMG ---
        exponent = lam*(mu(t) - r(t)) + 0.5*lam^2*sigma.^2;
        x_emg = e - lam*sigma;
        log_Phi = log(max(normcdf(x_emg), 1e-300));
        log_f1 = log_lam + exponent + log_Phi;

        % --- Mixture log-density ---
        log_a = log_1mpjt + log_f0;
        log_b = log_pjt   + log_f1;
        cmax = max(log_a, log_b);
        logw = cmax + log(exp(log_a - cmax) + exp(log_b - cmax));
        logw(~isfinite(cmax)) = -1e10;

        c = max(logw);
        w = exp(logw - c);
        sw = sum(w);
        if ~(isfinite(sw) && sw>0)
            nll = 1e8 + 1e4*sum(theta.^2);
            out = struct('h_mean', zeros(T,1), 'h_var', zeros(T,1), ...
                         'ess', zeros(T,1), 'pj_path', zeros(T,1));
            return;
        end
        w = w / sw;

        loglik = loglik + (c + log(sw) - log(N));

        ess = 1/sum(w.^2);
        ess_path(t) = ess;
        h_mean(t) = sum(w .* h);
        h_var(t)  = sum(w .* (h - h_mean(t)).^2);

        if ess < pf.resample_thresh * N
            idx = systematic_resample(w);
            h = h(idx);
        end
    end

    nll = -loglik;
    out = struct();
    out.h_mean  = h_mean;
    out.h_var   = h_var;
    out.ess     = ess_path;
    out.pj_path = pj_path;
end

%% ============================================================
%  Systematic resampling
%% ============================================================
function idx = systematic_resample(w)
    N = length(w);
    cdf = cumsum(w);
    u0 = rand()/N;
    u = u0 + (0:N-1)'/N;

    idx = zeros(N,1);
    i = 1;
    for j = 1:N
        while u(j) > cdf(i)
            i = i + 1;
        end
        idx(j) = i;
    end
end

%% ============================================================
%  Numerical Hessian (central differences)
%% ============================================================
function H = numhess(f, x0, h)
    x0 = x0(:);
    n = length(x0);
    H = zeros(n,n);

    for i = 1:n
        ei = zeros(n,1); ei(i) = 1;
        for j = i:n
            ej = zeros(n,1); ej(j) = 1;
            if i==j
                fpp = f(x0 + h*ei);
                fmm = f(x0 - h*ei);
                f00 = f(x0);
                H(i,i) = (fpp - 2*f00 + fmm) / (h^2);
            else
                fpp = f(x0 + h*ei + h*ej);
                fpm = f(x0 + h*ei - h*ej);
                fmp = f(x0 - h*ei + h*ej);
                fmm = f(x0 - h*ei - h*ej);
                hij = (fpp - fpm - fmp + fmm) / (4*h^2);
                H(i,j) = hij;
                H(j,i) = hij;
            end
        end
    end
end

%% ============================================================
%  Parameter packing/unpacking
%  theta = [alpha; beta; gamma(K); omega; eta_phi; eta_sh;
%           delta(K); psi0; psi(Kv); eta_muj]
%% ============================================================
function theta = pack_params(alpha, beta, gamma, omega, phi, sigmah, ...
                              delta, psi0, psi, muj)
    eta_phi = atanh(max(min(phi,0.999),-0.999));
    eta_sh  = log(max(sigmah, 1e-6));
    eta_muj = log(max(muj - 0.01, 1e-6));
    theta = [alpha; beta; gamma(:); omega; eta_phi; eta_sh; ...
             delta(:); psi0; psi(:); eta_muj];
end

function p = unpack_params(theta, K, Kv)
    idx = 0;
    p = struct();

    idx = idx + 1; p.alpha = theta(idx);
    idx = idx + 1; p.beta  = theta(idx);

    if K > 0
        p.gamma = theta(idx+1:idx+K); idx = idx + K;
    else
        p.gamma = zeros(0,1);
    end

    idx = idx + 1; p.omega = theta(idx);

    idx = idx + 1; eta_phi = theta(idx);
    p.phi = tanh(eta_phi);

    idx = idx + 1; eta_sh = theta(idx);
    p.sigmah = exp(eta_sh);

    if K > 0
        p.delta = theta(idx+1:idx+K); idx = idx + K;
    else
        p.delta = zeros(0,1);
    end

    idx = idx + 1; p.psi0 = theta(idx);

    if Kv > 0
        p.psi = theta(idx+1:idx+Kv); idx = idx + Kv;
    else
        p.psi = zeros(0,1);
    end

    idx = idx + 1; eta_muj = theta(idx);
    p.mu_j = exp(eta_muj);
end
