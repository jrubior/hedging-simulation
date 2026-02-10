function save_smm_params(results, filename)
%SAVE_SMM_PARAMS  Write calibrated SMM parameters to a text file.
%
%   save_smm_params(results, filename)
%
%   Writes all fields of results.param_hat plus the standardization
%   constants (S_mean, S_std, V_mean, V_std) to a plain-text file.
%   Format: one parameter per line, key followed by space-separated values.

    p = results.param_hat;

    fid = fopen(filename, 'w');
    if fid == -1
        error('save_smm_params:fileOpen', 'Cannot open %s for writing.', filename);
    end

    write_line(fid, 'alpha',   p.alpha);
    write_line(fid, 'beta',    p.beta);
    write_line(fid, 'gamma',   p.gamma);
    write_line(fid, 'omega',   p.omega);
    write_line(fid, 'phi',     p.phi);
    write_line(fid, 'sigmah',  p.sigmah);
    write_line(fid, 'delta_v', p.delta_v);
    write_line(fid, 'nu',      p.nu);
    write_line(fid, 'lambda',  p.lambda);
    write_line(fid, 'rho',     p.rho);
    write_line(fid, 'mu_j',    p.mu_j);
    write_line(fid, 'psi0',    p.psi0);
    write_line(fid, 'psi',     p.psi);
    write_line(fid, 'S_mean',  results.S_mean);
    write_line(fid, 'S_std',   results.S_std);
    write_line(fid, 'V_mean',  results.V_mean);
    write_line(fid, 'V_std',   results.V_std);

    fclose(fid);
    fprintf('Parameters saved to %s\n', filename);
end

function write_line(fid, name, vals)
    fprintf(fid, '%s', name);
    for i = 1:numel(vals)
        fprintf(fid, ' %.15g', vals(i));
    end
    fprintf(fid, '\n');
end
