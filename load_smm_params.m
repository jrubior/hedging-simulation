function results = load_smm_params(filename)
%LOAD_SMM_PARAMS  Read calibrated SMM parameters from a text file.
%
%   results = load_smm_params(filename)
%
%   Reads a plain-text parameter file written by save_smm_params and
%   returns a struct with fields:
%     results.param_hat  - struct of model parameters
%     results.S_mean, results.S_std  - state standardization constants
%     results.V_mean, results.V_std  - volatility standardization constants

    fid = fopen(filename, 'r');
    if fid == -1
        error('load_smm_params:fileOpen', 'Cannot open %s for reading.', filename);
    end

    kv = struct();
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if isempty(line) || line(1) == '#'
            continue;
        end
        tokens = strsplit(line);
        key = tokens{1};
        vals = str2double(tokens(2:end));
        kv.(key) = vals;
    end
    fclose(fid);

    % Build param_hat struct
    p = struct();
    p.alpha   = kv.alpha;
    p.beta    = kv.beta;
    p.gamma   = kv.gamma(:);
    p.omega   = kv.omega;
    p.phi     = kv.phi;
    p.sigmah  = kv.sigmah;
    p.delta_v = kv.delta_v;
    p.nu      = kv.nu;
    p.lambda  = kv.lambda;
    p.rho     = kv.rho;
    p.mu_j    = kv.mu_j;
    p.psi0    = kv.psi0;
    p.psi     = kv.psi(:);

    results = struct();
    results.param_hat = p;
    results.S_mean    = kv.S_mean;
    results.S_std     = kv.S_std;
    results.V_mean    = kv.V_mean;
    results.V_std     = kv.V_std;

    fprintf('Parameters loaded from %s\n', filename);
end
