
clc; clear; close all;

%% ── 1. LOAD MODEL ────────────────────────────────────────
load('inversion_model/saved_weights/inverse_pinn_model.mat');
% Loads: net, Xmu, Xsig, Ymu, Ysig, paramNames, PREDICT_VP

fprintf('Model loaded. Input size: %d | Output size: %d\n', ...
    numel(Xmu), numel(Ymu));

%% ── 2. LOAD TEST DATA ────────────────────────────────────
data = readtable('inversion_model/data/testingdata.xlsx');
sample_ids = data.sample_id;
unique_ids = unique(sample_ids);
numSamples = numel(unique_ids);
fprintf('Found %d test samples in testingdata.xlsx\n', numSamples);

%% ── 3. BUILD X AND Y  ────────────────────────────────────
%  X : [freq, Vr]               — same 2-column format as training
%  Y : [vs1, vs2, den1, den2, h1] — same 5-column targets

X_raw = [data.freq, data.Vr];

if PREDICT_VP
    Y_raw = [data.vp1, data.vp2, data.vs1, data.vs2, ...
             data.den1, data.den2, data.h1];
else
    Y_raw = [data.vs1, data.vs2, data.den1, data.den2, data.h1];
end
nOut = numel(paramNames);

%% ── 4. NORMALISE USING TRAINING STATS ───────────────────
Xn = (X_raw - Xmu) ./ Xsig;   % normalise all 1600 rows

%% ── 5. PREDICT ───────────────────────────────────────────
dlXn    = dlarray(Xn', 'CB');                        % [2 × 1600]
Ypred_n = extractdata(forward(net, dlXn))';          % [1600 × nOut]
Ypred   = Ypred_n .* Ysig + Ymu;                    % denormalise
Ytrue   = Y_raw;

%% ── 6. ROW-LEVEL METRICS (all 1600 rows) ────────────────
rmse_row   = sqrt(mean((Ypred - Ytrue).^2, 1));
relErr_row = 100 * rmse_row ./ mean(Ytrue, 1);
R2_row     = 1 - sum((Ypred-Ytrue).^2,1) ./ ...
                 sum((Ytrue-mean(Ytrue,1)).^2, 1);
mae_row    = mean(abs(Ypred - Ytrue), 1);

fprintf('\n══ Row-Level Test Results (all %d rows) ══════════════════\n', ...
    size(Ytrue,1));
fprintf('%-10s  %10s  %12s  %10s  %8s\n', ...
    'Parameter','RMSE','RelErr (%)','MAE','R²');
fprintf('%s\n', repmat('-',1,58));
for k = 1:nOut
    fprintf('%-10s  %10.4f  %12.2f  %10.4f  %8.4f\n', ...
        paramNames{k}, rmse_row(k), relErr_row(k), mae_row(k), R2_row(k));
end
fprintf('%s\n', repmat('-',1,58));
fprintf('%-10s  %10.4f  %12.2f  %10.4f  %8.4f\n', 'MEAN', ...
    mean(rmse_row), mean(relErr_row), mean(mae_row), mean(R2_row));

%% ── 7. SAMPLE-LEVEL METRICS ──────────────────────────────
%  Average predictions across 80 frequencies per sample
%  (each sample has one true parameter set — take median of predictions)
Ypred_sample = zeros(numSamples, nOut);
Ytrue_sample = zeros(numSamples, nOut);

for s = 1:numSamples
    mask = (sample_ids == unique_ids(s));
    Ypred_sample(s,:) = median(Ypred(mask,:), 1);   % median over 80 freq rows
    Ytrue_sample(s,:) = Y_raw(find(mask,1), :);      % same for all rows
end

rmse_s   = sqrt(mean((Ypred_sample - Ytrue_sample).^2, 1));
relErr_s = 100 * rmse_s ./ mean(Ytrue_sample, 1);
R2_s     = 1 - sum((Ypred_sample-Ytrue_sample).^2,1) ./ ...
               sum((Ytrue_sample-mean(Ytrue_sample,1)).^2,1);

fprintf('\n══ Sample-Level Results (%d samples, median aggregation) ══\n', numSamples);
fprintf('%-10s  %10s  %12s  %8s\n','Parameter','RMSE','RelErr (%)','R²');
fprintf('%s\n', repmat('-',1,45));
for k = 1:nOut
    fprintf('%-10s  %10.4f  %12.2f  %8.4f\n', ...
        paramNames{k}, rmse_s(k), relErr_s(k), R2_s(k));
end
fprintf('%s\n', repmat('-',1,45));
fprintf('%-10s  %10.4f  %12.2f  %8.4f\n','MEAN', ...
    mean(rmse_s), mean(relErr_s), mean(R2_s));

%% ── 8. SAVE METRICS TABLE ────────────────────────────────
T = table(paramNames', rmse_s', relErr_s', R2_s', ...
    'VariableNames',{'Parameter','RMSE','RelativeError_pct','R2'});
disp(T);
writetable(T, 'validation_metrics.csv');
fprintf('Metrics saved → validation_metrics.csv\n');

%% ── 9. PARITY PLOTS (sample-level) ──────────────────────
figure('Name','True vs Predicted','Position',[100 100 1300 600]);
nCols = ceil(nOut/2);
for k = 1:nOut
    subplot(2, nCols, k);
    scatter(Ytrue_sample(:,k), Ypred_sample(:,k), 80, 'filled', ...
        'MarkerFaceAlpha',0.75, 'MarkerFaceColor',[0.2 0.5 0.8]);
    hold on;
    mn  = min([Ytrue_sample(:,k); Ypred_sample(:,k)]);
    mx  = max([Ytrue_sample(:,k); Ypred_sample(:,k)]);
    pad = 0.05*(mx-mn);
    plot([mn-pad mx+pad],[mn-pad mx+pad],'r--','LineWidth',1.8);
    xlabel(['True ' paramNames{k}], 'FontSize',9);
    ylabel(['Pred ' paramNames{k}], 'FontSize',9);
    title(sprintf('%s\nRMSE=%.2f  R²=%.3f', ...
        paramNames{k}, rmse_s(k), R2_s(k)), 'FontSize',9);
    xlim([mn-pad mx+pad]); ylim([mn-pad mx+pad]);
    axis equal; grid on;
end
if nOut < nCols*2
    subplot(2, nCols, nOut+1); axis off;
end
sgtitle(sprintf('Inverse PINN Validation: True vs Predicted  (%d samples)', numSamples), ...
    'FontSize',12,'FontWeight','bold');
saveas(gcf,'validation_parity_plots.png');
fprintf('Parity plots saved → validation_parity_plots.png\n');

%% ── 10. PREDICTION vs FREQUENCY CURVES ──────────────────
%  For each sample, plot how each predicted parameter varies
%  across 80 frequencies (should be ~flat if model is consistent)
figure('Name','Prediction Stability Across Frequency','Position',[100 100 1300 700]);
colors = lines(numSamples);
for k = 1:nOut
    subplot(2, nCols, k); hold on;
    for s = 1:numSamples
        mask  = (sample_ids == unique_ids(s));
        freqs = data.freq(mask);
        preds = Ypred(mask, k);
        [fs, si] = sort(freqs);
        plot(fs, preds(si), 'Color', [colors(s,:) 0.6], 'LineWidth', 1);
        % True value as horizontal dashed line
        yline(Ytrue_sample(s,k), '--', 'Color', colors(s,:), ...
            'LineWidth', 0.8, 'Alpha', 0.5);
    end
    xlabel('Frequency (Hz)','FontSize',8);
    ylabel(paramNames{k},'FontSize',8);
    title(paramNames{k},'FontSize',9);
    grid on;
end
if nOut < nCols*2
    subplot(2, nCols, nOut+1); axis off;
end
sgtitle('Predicted Parameter vs Frequency (solid=pred, dashed=true)', ...
    'FontSize',11,'FontWeight','bold');
saveas(gcf,'validation_frequency_curves.png');
fprintf('Frequency curve plots saved → validation_frequency_curves.png\n');

%% ── 11. ERROR BAR CHART ──────────────────────────────────
figure('Name','Error Summary','Position',[100 100 900 400]);
subplot(1,2,1);
bar(rmse_s,'FaceColor',[0.2 0.5 0.8]);
set(gca,'XTickLabel',paramNames,'XTick',1:nOut,'FontSize',9);
ylabel('RMSE'); title('RMSE per Parameter'); grid on;

subplot(1,2,2);
bar(relErr_s,'FaceColor',[0.9 0.4 0.2]);
set(gca,'XTickLabel',paramNames,'XTick',1:nOut,'FontSize',9);
ylabel('Relative Error (%)'); title('Relative Error per Parameter'); grid on;
sgtitle('Validation Error Summary','FontSize',11,'FontWeight','bold');
saveas(gcf,'validation_error_summary.png');
fprintf('Error summary saved → validation_error_summary.png\n');

%% ── 12. PER-SAMPLE CONSOLE SUMMARY ──────────────────────
fprintf('\n══ Per-Sample Detail ════════════════════════════════════════\n');
header = sprintf('%-6s', 'Smpl');
for k=1:nOut; header = [header sprintf('  %16s', paramNames{k})]; end
fprintf('%s\n', header);
fprintf('%s\n', repmat('-', 1, 6 + nOut*18));

for s = 1:numSamples
    rowT = sprintf('%-6s', sprintf('S%d_T', unique_ids(s)));
    rowP = sprintf('%-6s', sprintf('S%d_P', unique_ids(s)));
    rowE = sprintf('%-6s', sprintf('S%d_%%', unique_ids(s)));
    for k = 1:nOut
        rowT = [rowT sprintf('  %16.4f', Ytrue_sample(s,k))];
        rowP = [rowP sprintf('  %16.4f', Ypred_sample(s,k))];
        err  = 100*abs(Ypred_sample(s,k)-Ytrue_sample(s,k))/abs(Ytrue_sample(s,k));
        rowE = [rowE sprintf('  %15.2f%%', err)];
    end
    fprintf('%s\n%s\n%s\n', rowT, rowP, rowE);
    fprintf('%s\n', repmat('-', 1, 6 + nOut*18));
end

fprintf('\nValidation complete.\n');