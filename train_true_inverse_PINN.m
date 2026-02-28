%  INVERSE PINN — Rayleigh Wave Dispersion Inversion
%  Two-Layer Earth Model
%
%  Input  : [freq (Hz),  Vr (m/s)]          — 2 observables
%  Output : [vs1, vs2, den1, den2, h1]       — 5 unknowns
%           (vp1, vp2 also predicted → 7 total if uncomment flag)
%
%  Loss = lambda_d * L_data
%       + lambda_p * L_physics      (F(w,Vr,params)=0)
%       + lambda_m * L_monotonic    (vs2 > vs1 soft constraint)
%
%  Dataset : pinns_dataset_20samples.xlsx
%            Columns: sample_id|freq|Vr|vp1|vp2|vs1|vs2|den1|den2|h1
clc; clear; close all;

PREDICT_VP = false;   % true  → 7 outputs (vp1,vp2,vs1,vs2,den1,den2,h1)
                      % false → 5 outputs (vs1,vs2,den1,den2,h1)  ← your request

%%  1. LOAD DATA 
data = readtable('data/pinns_dataset_20samples.xlsx');
% Columns: sample_id | freq | Vr | vp1 | vp2 | vs1 | vs2 | den1 | den2 | h1

sample_ids = data.sample_id;
unique_ids = unique(sample_ids);
numSamples = numel(unique_ids);

%  Inputs: frequency + Vr (the 2 observables)
X_raw = [data.freq, data.Vr];          % [N×2]

%  Targets: subsurface parameters
if PREDICT_VP
    Y_raw = [data.vp1, data.vp2, data.vs1, data.vs2, data.den1, data.den2, data.h1];           % [N×7]
    paramNames = {'vp1','vp2','vs1','vs2','den1','den2','h1'};
else
    Y_raw = [data.vs1, data.vs2, data.den1, data.den2, data.h1]; % [N×5]
    paramNames = {'vs1','vs2','den1','den2','h1'};
end
nOut = size(Y_raw, 2);

% Keep vp1,vp2 for physics residual even if not predicting them
vp_all = [data.vp1, data.vp2];   % [N×2]

fprintf('Loaded %d rows | Inputs: [freq, Vr] | Outputs: %d params\n', ...
    height(data), nOut);

%%  2. SAMPLE-WISE TRAIN / VAL / TEST SPLIT 
rng(42);
perm   = randperm(numSamples);
nTrain = floor(0.70 * numSamples);
nVal   = floor(0.15 * numSamples);
trainSamples = unique_ids(perm(1:nTrain));
valSamples   = unique_ids(perm(nTrain+1:nTrain+nVal));
testSamples  = unique_ids(perm(nTrain+nVal+1:end));

trMask = ismember(sample_ids, trainSamples);
vaMask = ismember(sample_ids, valSamples);
teMask = ismember(sample_ids, testSamples);

Xtrain = X_raw(trMask,:);  Ytrain = Y_raw(trMask,:);  VPtrain = vp_all(trMask,:);
Xval   = X_raw(vaMask,:);  Yval   = Y_raw(vaMask,:);
Xtest  = X_raw(teMask,:);  Ytest  = Y_raw(teMask,:);

fprintf('Split: %d train | %d val | %d test rows\n', ...
    size(Xtrain,1), size(Xval,1), size(Xtest,1));

%%  3. NORMALISATION 
Xmu  = mean(Xtrain,1);   Xsig = std(Xtrain,0,1);  Xsig(Xsig==0)=1;
Ymu  = mean(Ytrain,1);   Ysig = std(Ytrain,0,1);  Ysig(Ysig==0)=1;

norm_X  = @(X) (X - Xmu)  ./ Xsig;
norm_Y  = @(Y) (Y - Ymu)  ./ Ysig;
denorm_Y = @(Yn) Yn .* Ysig + Ymu;

Xtrain_n = norm_X(Xtrain);  Ytrain_n = norm_Y(Ytrain);
Xval_n   = norm_X(Xval);    Yval_n   = norm_Y(Yval);
Xtest_n  = norm_X(Xtest);

%% 4. COLLOCATION POINTS 
%  Use training rows as collocation points.
%  For physics residual we need vp (to complete the model), so we
%  pass VPtrain alongside. If predicting vp, we use predicted vp instead.
numColloc = 1500;
rng(7);
cIdx        = randi(size(Xtrain,1), numColloc, 1);
Xcolloc     = Xtrain(cIdx,:);       % [freq, Vr] original scale
Xcolloc_n   = Xtrain_n(cIdx,:);     % normalised
VPcolloc    = VPtrain(cIdx,:);      % [vp1, vp2] — used if PREDICT_VP=false
Ytrue_colloc= Ytrain(cIdx,:);       % true params at collocation points

%%  5. NETWORK ─
%  2 → 128 → 128 → 64 → nOut
%  tanh activations (smooth gradients, standard for PINNs)
layers = [
    featureInputLayer(2,      'Name','in')
    fullyConnectedLayer(128,  'Name','fc1')
    tanhLayer(                'Name','tanh1')
    fullyConnectedLayer(128,  'Name','fc2')
    tanhLayer(                'Name','tanh2')
    fullyConnectedLayer(64,   'Name','fc3')
    tanhLayer(                'Name','tanh3')
    fullyConnectedLayer(nOut, 'Name','out')
];
net = dlnetwork(layers);
nParams = sum(cellfun(@numel, net.Learnables.Value));
fprintf('Network: 2→128→128→64→%d  (%d parameters)\n', nOut, nParams);

%%  6. HYPERPARAMETERS ─
epochs    = 6000;
lr        = 1e-3;
batchSize = 256;

% Loss weights
lambda_d = 1.0;    % data (supervised) loss
lambda_p = 0.5;    % physics residual  F(w,Vr,params)=0
lambda_m = 0.1;    % monotonicity: vs2 > vs1

%%  7. DLARRAYS 
dlXtr  = dlarray(Xtrain_n', 'CB');
dlYtr  = dlarray(Ytrain_n', 'CB');
dlXva  = dlarray(Xval_n',   'CB');
dlYva  = dlarray(Yval_n',   'CB');
dlXco  = dlarray(Xcolloc_n','CB');

%%  8. TRAINING LOOP ─
avgG = [];  avgSqG = [];
bestValLoss = inf;   bestNet = net;

histTrain = nan(epochs,1);
histVal   = nan(floor(epochs/100),1);
logIdx    = 0;

fprintf('\n%-8s  %-12s  %-12s  %-12s  %-12s\n', ...
    'Epoch','Total','DataLoss','PhysLoss','ValLoss');
fprintf('%s\n', repmat('-',1,62));

for ep = 1:epochs

    % Mini-batch
    nTr  = size(Xtrain_n,1);
    bIdx = randperm(nTr, min(batchSize,nTr));
    dlXb = dlarray(Xtrain_n(bIdx,:)', 'CB');
    dlYb = dlarray(Ytrain_n(bIdx,:)', 'CB');

    % Collocation: subsample each epoch for speed
    coIdx = randperm(numColloc, min(500,numColloc));
    dlXco_ep = dlarray(Xcolloc_n(coIdx,:)', 'CB');
    Xco_ep   = Xcolloc(coIdx,:);
    VPco_ep  = VPcolloc(coIdx,:);

    % Forward + gradients
    [loss, grads] = dlfeval(@inversePINNLoss, net, ...
        dlXb, dlYb, dlXco_ep, Xco_ep, VPco_ep, ...
        Xmu, Xsig, Ymu, Ysig, ...
        lambda_d, lambda_p, lambda_m, PREDICT_VP);

    [net, avgG, avgSqG] = adamupdate(net, grads, avgG, avgSqG, ep, lr);
    histTrain(ep) = extractdata(loss);

    % Validation log
    if mod(ep,100)==0
        logIdx = logIdx+1;
        YpV   = forward(net, dlXva);
        vLoss = mean((YpV - dlYva).^2,'all');
        histVal(logIdx) = extractdata(vLoss);

        % Component losses for display
        Yp_b   = forward(net, dlXb);
        dL     = mean((Yp_b - dlYb).^2,'all');
        pL     = physicsResidual(net, dlXco_ep, Xco_ep, VPco_ep, ...
                                 Xmu, Xsig, Ymu, Ysig, PREDICT_VP);

        fprintf('%-8d  %-12.4e  %-12.4e  %-12.4e  %-12.4e\n', ep, ...
            extractdata(loss), extractdata(dL), ...
            extractdata(pL), extractdata(vLoss));

        if extractdata(vLoss) < bestValLoss
            bestValLoss = extractdata(vLoss);
            bestNet     = net;
        end
    end
end

net = bestNet;
fprintf('\nBest validation loss: %.4e\n', bestValLoss);

%%  9. TEST EVALUATION ─
dlXte   = dlarray(Xtest_n', 'CB');
Ypred_n = extractdata(forward(net, dlXte))';
Ypred   = denorm_Y(Ypred_n);
Ytrue   = Ytest;

rmse    = sqrt(mean((Ypred - Ytrue).^2, 1));
relErr  = 100 * rmse ./ mean(Ytrue, 1);
SS_res  = sum((Ypred-Ytrue).^2, 1);
SS_tot  = sum((Ytrue - mean(Ytrue,1)).^2, 1);
R2      = 1 - SS_res./SS_tot;

fprintf('\n══ Test Results ══════════════════════════════════════════\n');
fprintf('%-10s  %10s  %14s  %8s\n','Parameter','RMSE','RelErr (%)','R²');
fprintf('%s\n', repmat('-',1,50));
for k = 1:nOut
    fprintf('%-10s  %10.4f  %14.2f  %8.4f\n', ...
        paramNames{k}, rmse(k), relErr(k), R2(k));
end
fprintf('%s\n', repmat('-',1,50));
fprintf('%-10s  %10.4f  %14.2f  %8.4f\n','MEAN', ...
    mean(rmse), mean(relErr), mean(R2));

%%  10. PLOTS 

% Loss history
figure('Name','Inverse PINN – Training History');
semilogy(1:epochs, histTrain, 'b','LineWidth',1.2); hold on;
semilogy(100:100:epochs, histVal(1:logIdx),'r--o','LineWidth',1.5,'MarkerSize',4);
xlabel('Epoch'); ylabel('Loss (MSE, normalised)');
title('Inverse PINN Training History');
legend('Train total','Validation','Location','northeast'); grid on;

% Per-parameter parity plots
figure('Name','Inverse PINN – Parity Plots');
nCols = ceil(nOut/2);
for k = 1:nOut
    subplot(2, nCols, k);
    scatter(Ytrue(:,k), Ypred(:,k), 20, 'filled', 'MarkerFaceAlpha',0.5);
    hold on;
    lims = [min([Ytrue(:,k);Ypred(:,k)])*0.97, max([Ytrue(:,k);Ypred(:,k)])*1.03];
    plot(lims,lims,'r--','LineWidth',1.5);
    xlabel(['True ' paramNames{k}]);
    ylabel(['Pred ' paramNames{k}]);
    title(sprintf('%s | RMSE=%.2f | R²=%.4f', paramNames{k}, rmse(k), R2(k)));
    grid on; axis equal; xlim(lims); ylim(lims);
end
sgtitle('Inverse PINN – Per-Parameter Parity Plots');

% RMSE bar chart
figure('Name','Inverse PINN – RMSE per Parameter');
bar(rmse, 'FaceColor',[0.2 0.5 0.8]);
set(gca,'XTickLabel',paramNames,'XTick',1:nOut);
ylabel('RMSE (original units)');
title('Inverse PINN – Test RMSE per Parameter');
grid on;

% Relative error bar chart
figure('Name','Inverse PINN – Relative Error');
bar(relErr,'FaceColor',[0.9 0.4 0.2]);
set(gca,'XTickLabel',paramNames,'XTick',1:nOut);
ylabel('Relative Error (%)');
title('Inverse PINN – Test Relative Error per Parameter');
grid on;

%%  11. SAVE ─
save('inversion_model/saved_weights/inverse_pinn_model.mat','net','Xmu','Xsig','Ymu','Ysig', ...
    'paramNames','lambda_d','lambda_p','lambda_m','PREDICT_VP');
fprintf('\nModel saved → inverse_pinn_model.mat\n');

%%  LOCAL FUNCTIONS
function [L, grads] = inversePINNLoss(net, dlXb, dlYb, ...
    dlXco, Xco, VPco, Xmu, Xsig, Ymu, Ysig, ...
    lambda_d, lambda_p, lambda_m, PREDICT_VP)

    % Data loss
    Ypred  = forward(net, dlXb);
    L_data = mean((Ypred - dlYb).^2,'all');

    % Physics residual loss
    L_phys = physicsResidual(net, dlXco, Xco, VPco, ...
                              Xmu, Xsig, Ymu, Ysig, PREDICT_VP);

    % Monotonicity loss: vs2 > vs1
    %   Outputs normalised: find which columns are vs1(col3 if no vp, col1 otherwise)
    Yp_co  = forward(net, dlXco);          % normalised predictions
    if PREDICT_VP
        vs1_n = Yp_co(3,:);   vs2_n = Yp_co(4,:);
    else
        vs1_n = Yp_co(1,:);   vs2_n = Yp_co(2,:);
    end
    % Penalise when vs1 >= vs2  (soft ReLU penalty)
    L_mono = mean(max(dlarray(zeros(1,size(vs1_n,2),'like',vs1_n)), ...
                      vs1_n - vs2_n + 0.05).^2, 'all');

    L     = lambda_d*L_data + lambda_p*L_phys + lambda_m*L_mono;
    grads = dlgradient(L, net.Learnables);
end

%%  Physics residual ─
%  The network predicts params from (freq,Vr).
%  We plug those predicted params BACK into the Rayleigh
%  dispersion equation F(w, Vr, params) = 0.
%  Any non-zero F is a physics violation → penalise.
function L_phys = physicsResidual(net, dlXco, Xco, VPco, ...
                                   Xmu, Xsig, Ymu, Ysig, PREDICT_VP)

    numC    = size(Xco,1);
    Yp_n    = forward(net, dlXco);               % normalised predictions [nOut × numC]
    Yp      = extractdata(Yp_n)' .* Ysig + Ymu;  % denormalised [numC × nOut]

    % Extract predicted parameters
    if PREDICT_VP
        vp1_p = Yp(:,1);  vp2_p = Yp(:,2);
        vs1_p = Yp(:,3);  vs2_p = Yp(:,4);
        den1_p= Yp(:,5);  den2_p= Yp(:,6);
        h1_p  = Yp(:,7);
    else
        % vp1,vp2 taken from known training data at collocation points
        vp1_p = VPco(:,1);  vp2_p = VPco(:,2);
        vs1_p = Yp(:,1);    vs2_p = Yp(:,2);
        den1_p= Yp(:,3);    den2_p= Yp(:,4);
        h1_p  = Yp(:,5);
    end

    % True freq and Vr at collocation points (original scale)
    freq_c = Xco(:,1);   % Hz
    Vr_c   = Xco(:,2);   % m/s (this is the OBSERVED Vr — fixed)

    F_vec = zeros(numC,1);
    for i = 1:numC
        w_i  = 2*pi * freq_c(i);
        vp_i = [vp1_p(i), vp2_p(i)];
        vs_i = [vs1_p(i), vs2_p(i)];
        den_i= [den1_p(i),den2_p(i)];
        h1_i = h1_p(i);
        Vr_i = Vr_c(i);

        F_vec(i) = rayleighResidual(w_i, Vr_i, vp_i, vs_i, den_i, h1_i);
    end

    % Normalise by RMS scale to keep O(1)
    F_scale = sqrt(mean(F_vec.^2)) + 1e-10;
    L_phys  = dlarray(mean((F_vec ./ F_scale).^2));
end

%%  Rayleigh dispersion characteristic function 
%  Fast-vector-transfer method (same as data-generation code).
%  Returns F = E(5,1). F=0 on the true dispersion curve.
function F = rayleighResidual(w, Vr, vp, vs, den, h1)

    n  = 2;
    kk = w / Vr;

    % Guard: Vr must be < min(vs) for surface waves; clamp to avoid NaN
    if Vr >= min(vs) || Vr <= 0
        F = 1e6;   % large penalty, not on dispersion curve
        return;
    end

    rp = sqrt(Vr^2./vp.^2 - 1 + 0i);
    rs = sqrt(Vr^2./vs.^2 - 1 + 0i);
    r  = 1 - Vr^2./(2*(vs.^2));
    g  = 1 - r;
    rr = rp.^2;
    s  = rs.^2;
    p  = rp .* kk .* h1;
    q  = rs .* kk .* h1;
    a  = cos(p);
    b  = cos(q);
    c  = sin(p) ./ rp;
    d  = sin(q) ./ rs;
    l  = ones(1, n);   % density ratio simplification; full: vs^2*den/(vs^2*den)

    % Half-space initialisation (layer 2)
    E       = zeros(5, n);
    E(1, n) = 0;
    E(2, n) = 1 + rp(n)*rs(n);
    E(3, n) = r(n) + rp(n)*rs(n);
    E(4, n) = rs(n)*(1-r(n))*1i;
    E(5, n) = -r(n)^2 - rp(n)*rs(n);

    % Propagate through layer 1
    for m = n-1:-1:1
        M1 = [1,        2,       0,     0,     -1;
              r(m),     1+r(m),  0,     0,     -1;
              0,        0,       g(m),  0,      0;
              0,        0,       0,     g(m),   0;
             -r(m)^2,  -2*r(m), 0,     0,      1];

        L_mat = [a(m)*b(m),            0,  -a(m)*d(m),          b(m)*c(m),         c(m)*d(m);
                 0,                    1,   0,                   0,                 0;
                 a(m)*d(m)*s(m),       0,   a(m)*b(m),           c(m)*d(m)*s(m),  -b(m)*c(m);
                -b(m)*c(m)*rr(m),      0,   c(m)*d(m)*rr(m),     a(m)*b(m),        a(m)*d(m);
                 c(m)*d(m)*rr(m)*s(m), 0,   b(m)*c(m)*rr(m),    -a(m)*d(m)*s(m),  a(m)*b(m)];

        M2 = [1/l(m),     -2,     0,     0,    -l(m);
             -r(m)/l(m),  1+r(m), 0,     0,     l(m);
              0,           0,     g(m),  0,      0;
              0,           0,     0,     g(m),   0;
             -r(m)^2/l(m), 2*r(m),0,     0,     l(m)];

        E(:,m) = M1 * L_mat * M2 * E(:,m+1);
    end

    F = real(E(5,1));
end
