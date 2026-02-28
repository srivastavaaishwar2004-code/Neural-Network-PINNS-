clc;clear;
project_root = fileparts(pwd);

N_samples = 100;


lhs     = lhsdesign(N_samples, 5);
vs1_arr = 150  + lhs(:,1)  * 100;
vs2_arr = 300  + lhs(:,2)  * 200;
den1_arr = 2000 + lhs(:,3) * 200;
den2_arr = 2000 + lhs(:,4) * 300;
h1_arr   = 5    + lhs(:,5)  * 20;

fprint("Generating %d PINNs tarinig samples...\n", N_samples);
results = cell(N_samples, 1);

parfor s= 1:N_samples
    fprint('\n=== Sample %d ===\n', s);

    vs1 = vs1_arr(s);
    vs2 = vs2_arr(s);
    vp1 = 1.7 * vs1;
    vp2 = 1.7 * vs2;
    den1 = den1_arr(s);
    den2 = den2_arr(s);
    h1 = h1_arr(s);

    vp = [vp1 vp2];
    vs = [vs1 vs2];
    den = [den1 den2];
    h = [h1];

    wmin = 1;
    wwmax = 80;

    [freq_clean, vr_clean] = two_layer_rayleigh_data_genartion_modified( ...
        vp, vs, den, h, wmin, wwmax);
    
    if isempty(freq_clean)
        warning('no root found for sample %d. skipping.' s);
        results{s} = [];
        continue; 
    end
    
    nrow   = length(freq_clean);
    sample_id = s * ones(nrow, 1);
     
    results{s} = [
        sample_id, ...
        freq_clean, ...
        vr_clean, ...
        vp1 * ones(nrow, 1), ...
        vp2 * ones(nrow, 1), ...
        vs1 * ones(nrow, 1), ...
        vs2 * ones(nrow, 1), ...
        den1 * ones(nrow, 1), ...
        den2 * ones(nrow, 1), ...
        h1 * ones(nrow, 1)];
end

% Assemble 
ALL_DATA = [];
for s = 1:N_samples
    if ~isempty(results{s})
        ALL_DATA = [ALL_DATA; results{s}];
    end
end

% Ensure the outputdirectory exists 
out_dir = 'inversion_model/data';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% Save
header = {'sample_id','freq','Vr','vp1','vp2','vs1','vs2','den1','den2','h1'};

writecell(ALL_DATA, out_dir, 'all_data.csv', 'writeheader', header,'Sheet',1,'Range', 'A1');
writematrix(ALL_DATA, fullfile(out_dir, 'all_data.xlsx'), 'Sheet', 1, 'Range', 'A2');
writematrix(ALL_DATA,fullfile(out_dir,'pinns_dataset.csv'));

fprintf('\nDataset generation completed successfully\n');
fprintf('Total rows written %d\n',size(ALL_DATA,1));
