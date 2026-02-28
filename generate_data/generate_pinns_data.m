5clc; clear;
project_root = fileparts(pwd);
addpath(genpath(project_root));

N_samples = 100;  

% Latin Hypercube Sampling over 5 independent parameters
lhs      = lhsdesign(N_samples, 5);
vs1_arr  = 150  + lhs(:,1) * 100;   % 150-250 m/s
vs2_arr  = 300  + lhs(:,2) * 200;   % 300-500 m/s
den1_arr = 1700 + lhs(:,3) * 200;   % 1700-1900 kg/m3
den2_arr = 1900 + lhs(:,4) * 300;   % 1900-2200 kg/m3
h1_arr   = 5    + lhs(:,5) * 20;    % 5-25 m

fprintf('Generating %d PINNs training samples...\n', N_samples);
results = cell(N_samples, 1);

parfor s = 1:N_samples
    fprintf('\n=== Sample %d ===\n', s);

    vs1  = vs1_arr(s);
    vs2  = vs2_arr(s);
    vp1  = 1.7 * vs1;
    vp2  = 1.7 * vs2;
    den1 = den1_arr(s);
    den2 = den2_arr(s);
    h1   = h1_arr(s);

    vp  = [vp1 vp2];
    vs  = [vs1 vs2];
    den = [den1 den2];
    h   = [h1];

    wwmin = 1;
    wwmax = 80;

    [freq_clean, vr_clean] = two_layer_rayleigh_data_generation_modified( ...
        vp, vs, den, h, wwmin, wwmax);

    if isempty(freq_clean)
        warning('No roots found for sample %d. Skipping.', s);
        results{s} = [];
        continue;
    end

    nrows     = length(freq_clean);
    sample_id = s * ones(nrows, 1);

    results{s} = [ ...
        sample_id            ...
        freq_clean           ...
        vr_clean             ...
        vp1  * ones(nrows,1) ...
        vp2  * ones(nrows,1) ...
        vs1  * ones(nrows,1) ...
        vs2  * ones(nrows,1) ...
        den1 * ones(nrows,1) ...
        den2 * ones(nrows,1) ...
        h1   * ones(nrows,1)];
end

% Assemble
ALL_DATA = [];
for s = 1:N_samples
    if ~isempty(results{s})
        ALL_DATA = [ALL_DATA; results{s}];
    end
end

% Ensure output directory exists  <-- the fix
out_dir = 'invesion_model/data';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% Save
header = {'sample_id','freq','Vr','vp1','vp2','vs1','vs2','den1','den2','h1'};

writecell(header,  fullfile(out_dir, 'pinns_dataset.xlsx'), 'Sheet', 1, 'Range', 'A1');
writematrix(ALL_DATA, fullfile(out_dir, 'pinns_dataset.xlsx'), 'Sheet', 1, 'Range', 'A2');
writematrix(ALL_DATA, fullfile(out_dir, 'pinns_dataset.csv'));

fprintf('\nDataset generation completed successfully!\n');
fprintf('Total rows written: %d\n', size(ALL_DATA, 1));