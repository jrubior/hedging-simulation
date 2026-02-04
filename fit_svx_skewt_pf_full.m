
function results = fit_svx_skewt_pf_full(r, m, S, varargin)
%FIT_SVX_SKEWT_PF_FULL  SV-X with Hansen's skewed-t + leverage (monthly).
%
% Model:
%   r_t = alpha + beta*m_t + gamma'*s_t + exp(0.5*h_t)*eps_t
%   eps_t ~ Hansen_skewt(nu, lambda)   (mean 0, variance 1)
%   h_t = omega + phi*(h_{t-1}-omega) + delta'*s_t
%           + sigma_h*(rho*eps_{t-1} + sqrt(1-rho^2)*xi_t)
%   xi_t ~ N(0,1)
%
% Differences vs fit_svx_t_pf_full:
%   - Hansen's skewed Student-t innovations (parameter lambda)
%   - Leverage effect: correlation rho between eps_t and h_{t+1}
%
% Usage:
%   results = fit_svx_skewt_pf_full(r, m, S);
%
% Optional name-value pairs:
%   'NParticles'      (default 5000)
%   'Seed'            (default 123)
%   'ComputeStdErr'   (default true)
%   'HessStep'        (default 1e-4)
%   'Display'         (default 'iter')
%   'MaxIter'         (default 300)
%   'ResampleThresh'  (default 0.5)
%
% ------------------------------------------------------------

    if nargin < 3, S = []; end
    r = r(:); m = m(:);
    T = length(r);
    if length(m) ~= T, error('m must have same length as r'); end
    if ~isempty(S) && size(S,1) ~= T, error('S must have T rows'); end
    K = size(S,2);

    % -------- Parse options --------
    pOpt = inputParser;
    pOpt.addParameter('NParticles', 5000);
    pOpt.addParameter('Seed', 123);
    pOpt.addParameter('ComputeStdErr', true);
    pOpt.addParameter('HessStep', 1e-4);
    pOpt.addParameter('Display', 'iter');
    pOpt.addParameter('MaxIter', 300);
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

    % -------- Initial values via OLS (mean only) --------
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
    nu0      = 8;
    lambda0  = 0;      % no skewness initially
    rho0     = 0;      % no leverage initially

    theta0 = pack_params(alpha0, beta0, gamma0, omega0, phi0, sigmah0, ...
                          delta0, nu0, lambda0, rho0);

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
        'MaxFunctionEvaluations', 6000,...
        'OptimalityTolerance', 1e-6,...
        'StepTolerance', 1e-8);

    obj = @(th) negloglik_pf(th, r, m, Sz, pf);
    [theta_hat, fval, exitflag, output] = fminunc(obj, theta0, opts);

    % -------- Decode parameters --------
    param_hat = unpack_params(theta_hat, K);

    % -------- Run PF at MLE to get filtered summaries --------
    [nll, pf_out] = negloglik_pf(theta_hat, r, m, Sz, pf);

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
        H = numhess(@(th) negloglik_pf(th, r, m, Sz, pf), theta_hat, hstep);
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

    % Pretty print
    print_report(results);
end

%% ============================================================
%  Compute report statistics (monthly -> annualized)
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

    alpha_model_m = p.alpha;
    alpha_model_ann = 12*p.alpha;
    beta_model = p.beta;

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

    stats.beta_model = beta_model;
    stats.beta_ols = beta_ols;

    stats.alpha_model_monthly = alpha_model_m;
    stats.alpha_model_annual = alpha_model_ann;
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

    fprintf('\n========== SV-X (Skewed-t + Leverage) REPORT ==========\n');
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
    fprintf('nu     :                     %8.6f\n', p.nu);
    fprintf('lambda (skewness):           %8.6f\n', p.lambda);
    fprintf('rho    (leverage):           %8.6f\n', p.rho);

    if ~isempty(p.gamma)
        fprintf('gamma (mean states):         [%s]\n', sprintf('%.6f ', p.gamma));
    end
    if ~isempty(p.delta)
        fprintf('delta (vol  states):         [%s]\n', sprintf('%.6f ', p.delta));
    end

    if ~isempty(res.se_theta)
        fprintf('\n--- Approx. std. errors (theta parametrization) ---\n');
        fprintf('Note: these are for the transformed theta vector.\n');
        fprintf('se(theta):                   [%s]\n', sprintf('%.6f ', res.se_theta));
    end
    fprintf('=======================================================\n\n');
end

%% ============================================================
%  Negative log-likelihood via bootstrap particle filter
%  (with Hansen's skewed-t and leverage)
%% ============================================================
function [nll, out] = negloglik_pf(theta, r, m, Sz, pf)
    [T, K] = size(Sz);
    p = unpack_params(theta, K);

    alpha  = p.alpha;
    beta   = p.beta;
    gamma  = p.gamma;
    omega  = p.omega;
    phi    = p.phi;
    sigmah = p.sigmah;
    delta  = p.delta;
    nu     = p.nu;
    lambda = p.lambda;
    rho    = p.rho;

    N = pf.N;
    rng(pf.seed, 'twister');

    mu = alpha + beta*m + (K>0)*(Sz*gamma);

    % Hansen's skewed-t constants
    [logbc, a_skt, b_skt] = hansen_skewt_constants(nu, lambda);

    % Initialize particles for h_1 (no leverage for first period)
    if T >= 1
        mean_h1 = omega + (K>0)*(Sz(1,:)*delta);
        h = mean_h1 + sqrt(pf.h0_var)*randn(N,1);
    else
        h = omega + sqrt(pf.h0_var)*randn(N,1);
    end

    % eps stores the standardized residual at each particle (for leverage)
    eps_particles = zeros(N,1);

    loglik = 0;
    ess_path = zeros(T,1);
    h_mean = zeros(T,1);
    h_var  = zeros(T,1);

    sqrt_1mrho2 = sqrt(max(1 - rho^2, 0));

    for t = 1:T
        if t > 1
            % Propagate h_t using leverage: rho*eps_{t-1} + sqrt(1-rho^2)*xi_t
            xi = randn(N,1);
            mean_ht = omega + phi*(h - omega) + (K>0)*(Sz(t,:)*delta);
            h = mean_ht + sigmah*(rho*eps_particles + sqrt_1mrho2*xi);
        end

        sigma = exp(0.5*h);
        e = (r(t) - mu(t)) ./ sigma;

        % Log-weights using Hansen's skewed-t density
        logw = log_hansen_skewt(e, nu, lambda, logbc, a_skt, b_skt) - log(sigma);

        c = max(logw);
        w = exp(logw - c);
        sw = sum(w);
        if ~(isfinite(sw) && sw>0)
            nll = 1e8 + 1e4*sum(theta.^2);
            out = struct('h_mean', zeros(T,1), 'h_var', zeros(T,1), 'ess', zeros(T,1));
            return;
        end
        w = w / sw;

        loglik = loglik + (c + log(sw) - log(N));

        ess = 1/sum(w.^2);
        ess_path(t) = ess;
        h_mean(t) = sum(w .* h);
        h_var(t)  = sum(w .* (h - h_mean(t)).^2);

        % Store standardized residuals for leverage at t+1
        eps_particles = e;

        if ess < pf.resample_thresh * N
            idx = systematic_resample(w);
            h = h(idx);
            eps_particles = eps_particles(idx);
        end
    end

    nll = -loglik;
    out = struct();
    out.h_mean = h_mean;
    out.h_var  = h_var;
    out.ess    = ess_path;
end

%% ============================================================
%  Hansen's skewed-t: precompute constants
%% ============================================================
function [logbc, a, b] = hansen_skewt_constants(nu, lambda)
    logc = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*log(pi*(nu-2));
    c = exp(logc);
    a = 4*lambda*c*(nu-2)/(nu-1);
    b = sqrt(max(1 + 3*lambda^2 - a^2, 1e-10));
    logbc = log(b) + logc;
end

%% ============================================================
%  Hansen's skewed-t log density (mean 0, variance 1)
%% ============================================================
function logf = log_hansen_skewt(z, nu, lambda, logbc, a, b)
    % s = 1-lambda when z < -a/b, else 1+lambda
    threshold = -a/b;
    s = (1 - lambda) .* (z < threshold) + (1 + lambda) .* (z >= threshold);
    y = (b*z + a) ./ s;
    logf = logbc - ((nu+1)/2) .* log(1 + y.^2 / (nu-2));
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
%  Parameter packing/unpacking with constraints
%% ============================================================
function theta = pack_params(alpha, beta, gamma, omega, phi, sigmah, delta, nu, lambda, rho)
    eta_phi    = atanh(max(min(phi,0.999),-0.999));
    eta_sh     = log(max(sigmah, 1e-6));
    eta_nu     = log(max(nu-2, 1e-6));
    eta_lambda = atanh(max(min(lambda,0.999),-0.999));
    eta_rho    = atanh(max(min(rho,0.999),-0.999));
    theta = [alpha; beta; gamma(:); omega; eta_phi; eta_sh; delta(:); eta_nu; eta_lambda; eta_rho];
end

function p = unpack_params(theta, K)
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

    idx = idx + 1; eta_nu = theta(idx);
    p.nu = 2 + exp(eta_nu);

    idx = idx + 1; eta_lambda = theta(idx);
    p.lambda = tanh(eta_lambda);

    idx = idx + 1; eta_rho = theta(idx);
    p.rho = tanh(eta_rho);
end
