%% ================= LHHW MODEL – 6 STRUCTURES =============================
clc; clear; close all;

%% -------------------- DATA -----------------------------------------------
data873 = [
0.0229 0.345 0.0610 0.0030 0.0410 0.017 0.428
0.0251 0.332 0.0570 0.0028 0.0380 0.019 0.449
0.0260 0.315 0.0510 0.0026 0.0340 0.021 0.483
0.0267 0.300 0.0460 0.0024 0.0310 0.024 0.502
0.0272 0.288 0.0430 0.0023 0.0290 0.025 0.530];

data923 = [
0.028189233 0.453118984 0.111422233 0.00955852 0.083340908 0.011095881 0.370288119
0.03107576  0.430835611 0.102969739 0.006103287 0.063064471 0.014502794 0.401747507
0.032100557 0.418148897 0.09711066  0.004365558 0.051726019 0.017848888 0.431944699
0.034882565 0.405673269 0.081418952 0.003378186 0.048226084 0.01990191  0.451735136
0.035676095 0.3714673   0.07130478  0.002847645 0.032301    0.022055437 0.480351639];

data973 = [
0.032314 0.560318 0.154000 0.006598 0.098660 0.004853 0.280071
0.037321 0.514669 0.082700 0.003649 0.084956 0.006591 0.357622
0.040180 0.469148 0.006170 0.051162 0.029298 0.011142 0.389116
0.043698 0.448535 0.001260 0.065170 0.037320 0.014562 0.391354
0.043978 0.420000 0.000106 0.065170 0.037320 0.017346 0.420000];

Tlist = [873.15; 923.15; 973.15];
alldata = {data873, data923, data973};

expRate = [];
for k = 1:3
    expRate = [expRate; alldata{k}(:,1)];
end

%% -------------------- CONSTANTS ------------------------------------------
R  = 8.314462618;
Tm = 923.15;

K_SRP_map = containers.Map([873.15 923.15 973.15], ...
                           [1.30E-07 3.25E-04 2.50E-02]);

%% -------------------- INITIAL GUESS & BOUNDS ------------------------------
x0 = [2.3e4; 4.6e4; -60; -20];
lb = [1e-6; 3e3; -85; -40];
ub = [1e12; 6e5; -10; 0];

opts = optimoptions('lsqnonlin',...
    'Display','iter',...
    'FiniteDifferenceType','central',...
    'MaxFunctionEvaluations',2e5,...
    'MaxIterations',2e3);

%% ===================== MODEL LOOP ========================================
modelNames = {
    'SS-both-diss'
    'SS-P-diss'
    'SS-W-diss'
    'DS-both-diss'
    'DS-P-diss'
    'DS-W-diss'
};

r_pred_all = cell(numel(modelNames),1);

for modelID = 1:numel(modelNames)

    fprintf('\n================ %s =================\n',modelNames{modelID});

    resid = @(x) residuals_all(x,Tlist,alldata,K_SRP_map,R,expRate,modelID);
    [xhat,resnorm,~,~,~,~,J] = lsqnonlin(resid,x0,lb,ub,opts);

    r_pred = model_pred(xhat,Tlist,alldata,K_SRP_map,R,modelID);
    r_pred_all{modelID} = r_pred;

    SST  = sum((expRate-mean(expRate)).^2);
    R2   = 1 - resnorm/SST;
    RMSE = sqrt(mean((r_pred-expRate).^2));

    fprintf('km   = %.3e\n',xhat(1));
    fprintf('Ei   = %.2f kJ/mol\n',xhat(2)/1000);
    fprintf('DS_w = %.2f J/mol/K\n',xhat(3));
    fprintf('DS_p = %.2f J/mol/K\n',xhat(4));
    fprintf('SSE  = %.4e\n',resnorm);
    fprintf('R2   = %.4f\n',R2);
    fprintf('RMSE = %.4e\n',RMSE);
end

%% ===================== PARITY PLOTS ======================================
figure;
for i = 1:numel(modelNames)
    subplot(3,2,i)
    scatter(expRate,r_pred_all{i},60,'filled'); hold on
    mn = min([expRate; r_pred_all{i}]);
    mx = max([expRate; r_pred_all{i}]);
    plot([mn mx],[mn mx],'k--','LineWidth',1.3)
    axis equal; grid on
    title(['Parity – ',modelNames{i}])
end

%% ===================== RESIDUAL PLOTS ====================================
figure;
for i = 1:numel(modelNames)
    subplot(3,2,i)
    plot(r_pred_all{i}-expRate,'o-'); hold on
    yline(0,'k--')
    grid on
    title(['Residuals – ',modelNames{i}])
end

%% ===================== FUNCTIONS =========================================
function res = residuals_all(x,Tlist,data,Kmap,R,expRate,modelID)
res = model_pred(x,Tlist,data,Kmap,R,modelID) - expRate;
end

function r_pred = model_pred(x,Tlist,data,Kmap,R,modelID)

km   = x(1); 
Ei   = x(2); 
DS_w = x(3); 
DS_p = x(4);

Tm = 923.15;

% ---- heats of adsorption (J/mol) ----
DH_P_mol  = -52.5e3;   % phenol molecular
DH_W_mol  = -20.0e3;   % water molecular
DH_P_diss = -52.5e3;    % -100e3 phenol dissociative
DH_W_diss = -20.0e3;   % -67.0e3 water dissociative

Kmap_keys = cell2mat(keys(Kmap));
r_pred = [];

for k = 1:numel(Tlist)

    T = Tlist(k); 
    d = data{k};

    [~,ix] = min(abs(Kmap_keys - T));
    K_eq = Kmap(Kmap_keys(ix));

    % ---------- Select heats based on model ----------
    switch modelID
        case 1  % SS both diss
            DH_p = DH_P_diss;
            DH_w = DH_W_diss;

        case 2  % SS P diss
            DH_p = DH_P_diss;
            DH_w = DH_W_mol;

        case 3  % SS W diss
            DH_p = DH_P_mol;
            DH_w = DH_W_diss;

        case 4  % DS both diss
            DH_p = DH_P_diss;
            DH_w = DH_W_diss;

        case 5  % DS P diss
            DH_p = DH_P_diss;
            DH_w = DH_W_mol;

        case 6  % DS W diss
            DH_p = DH_P_mol;
            DH_w = DH_W_diss;
    end

    % ---------- Temperature-dependent parameters ----------
    K_P = exp(DS_p/R - DH_p/R*(1/T - 1/Tm));
    K_W = exp(DS_w/R - DH_w/R*(1/T - 1/Tm));
    kS  = km * exp(-Ei/R*(1/T - 1/Tm));

    % ---------- Partial pressures ----------
    P_H2  = d(:,2); 
    P_CO2 = d(:,3);
    P_P   = d(:,6); 
    P_H2O = d(:,7);

    % ---------- Reversible driving force ----------
    driving = (P_P .* P_H2O).^0.5 ...
        - (P_CO2.^6 .* P_H2.^14) ./ K_eq;

    % ---------- Rate expressions ----------
    switch modelID
        case 1  % SS both diss
            denom = (1 + (K_P.*P_P).^0.5 + (K_W.*P_H2O).^0.5).^2;
            rate  = kS*(K_P.*K_W).^0.5 .* driving ./ denom;

        case 2  % SS P diss
            denom = (1 + (K_P.*P_P).^0.5 + K_W.*P_H2O).^2;
            rate  = kS*(K_P).^0.5 .* K_W .* ((P_P).^0.5 .* P_H2O) ./ denom;

        case 3  % SS W diss
            denom = (1 + K_P.*P_P + (K_W.*P_H2O).^0.5).^2;
            rate  = kS*K_P .* (K_W).^0.5 .* (P_P .* (P_H2O).^0.5) ./ denom;

        case 4  % DS both diss
            denom = (1 + (K_P.*P_P).^0.5) .* (1 + (K_W.*P_H2O).^0.5);
            rate  = kS*(K_P.*K_W).^0.5 .* driving ./ denom;

        case 5  % DS P diss
            denom = (1 + (K_P.*P_P).^0.5) .* (1 + K_W.*P_H2O);
            rate  = kS*(K_P).^0.5 .* K_W .* ((P_P).^0.5 .* P_H2O) ./ denom;

        case 6  % DS W diss
            denom = (1 + K_P.*P_P) .* (1 + (K_W.*P_H2O).^0.5);
            rate  = kS*K_P .* (K_W).^0.5 .* (P_P .* (P_H2O).^0.5) ./ denom;
    end

    r_pred = [r_pred; rate];
end
end
