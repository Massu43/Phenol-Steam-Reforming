%% ============ LHHW CI + t-test + F-test (8 MODELS) ======================
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

Tlist   = [873.15; 923.15; 973.15];
alldata = {data873,data923,data973};

expRate=[];
for k=1:3
    expRate=[expRate;alldata{k}(:,1)];
end

%% -------------------- CONSTANTS ------------------------------------------
R=8.314462618;  Tm=923.15;
DH_p=-52.5e3;   DH_w=-20e3;

K_SRP_map=containers.Map([873.15 923.15 973.15],...
                         [1.30E-07 3.25E-04 2.50E-02]);

%% -------------------- BOUNDS ---------------------------------------------
LB_ALL=[1e-6;3e3;-85;-40];
UB_ALL=[1e12;6e5;-10;0];

%% -------------------- FIXED PARAMETER VALUES -----------------------------
fixedParams = struct();

fixedParams.SS_both_diss = [7.135e2;  96.66e3; -84.51; -39.28];
fixedParams.SS_P_diss    = [1.202e4; 106.50e3; -74.67; -11.39];
fixedParams.SS_W_diss    = [1.126e4; 139.77e3; -67.82; -34.43];
fixedParams.SS_nondiss   = [2.151e4; 156.84e3; -56.52; -14.18];

fixedParams.DS_both_diss = [7.017e2;  97.85e3; -84.51; -39.55];
fixedParams.DS_P_diss    = [1.479e4; 113.74e3; -72.46; -22.41];
fixedParams.DS_W_diss    = [1.179e4; 140.22e3; -70.52; -33.66];
fixedParams.DS_nondiss   = [2.685e4; 157.23e3; -56.07; -16.53];

modelNames = fieldnames(fixedParams);

%% -------------------- USER INPUT -----------------------------------------
fprintf('\nParameter indices:\n');
fprintf('1 = km,  2 = Ei,  3 = DS_w,  4 = DS_p\n');
fitIdx = input('Enter parameter indices to FIT (e.g. [1 2]): ');

opts=optimoptions('lsqnonlin','Display','iter','FiniteDifferenceType','central');

%% -------------------- PRECOMPUTE SST -------------------------------------
SST=sum((expRate-mean(expRate)).^2);
N=length(expRate);

%% ===================== MODEL LOOP ========================================
for m=1:numel(modelNames)

    modelName=modelNames{m};
    fprintf('\n================ %s =================\n',modelName);

    p_full=fixedParams.(modelName);
    x0=p_full(fitIdx);
    lb=LB_ALL(fitIdx); ub=UB_ALL(fitIdx);

    resid=@(x) residuals_general(x,p_full,fitIdx,modelName,...
        Tlist,alldata,K_SRP_map,R,Tm,DH_p,DH_w,expRate);

    [xhat,SSE,~,~,~,~,J]=lsqnonlin(resid,x0,lb,ub,opts);

    %% ---- CI + t-test ----
    p=length(xhat); dof=N-p;
    sigma2=SSE/dof;
    Cov=sigma2*inv(J'*J);
    se=sqrt(diag(Cov));
    tval=tinv(0.975,dof);

    CI_L=xhat-tval*se; CI_U=xhat+tval*se;

    for i=1:p
        fprintf('Param %d CI: [%.3g , %.3g]\n',fitIdx(i),CI_L(i),CI_U(i));
        fprintf('  t = %.2f (t_crit = %.2f)\n',xhat(i)/se(i),tval);
    end

    %% ---- GLOBAL F-TEST ----
    F_model=((SST-SSE)/p)/(SSE/(N-p));
    F_crit=finv(0.95,p,N-p);
    fprintf('Global F = %.2f (Fcrit = %.2f)\n',F_model,F_crit);
end

%% ===================== FUNCTIONS =========================================
function res=residuals_general(x,p_full,fitIdx,modelName,...
    Tlist,data,Kmap,R,Tm,DH_p,DH_w,expRate)

p=p_full; p(fitIdx)=x;
r=model_general(p,modelName,Tlist,data,Kmap,R,Tm,DH_p,DH_w);
res=r-expRate;
end

function r_pred=model_general(p,modelName,Tlist,data,Kmap,R,Tm,DH_p,DH_w)

km=p(1); Ei=p(2); DS_w=p(3); DS_p=p(4);
keysK=cell2mat(keys(Kmap)); r_pred=[];

for k=1:numel(Tlist)
    T=Tlist(k); d=data{k};
    [~,ix]=min(abs(keysK-T)); Keq=Kmap(keysK(ix));

    KP=exp(DS_p/R - DH_p/R*(1/T-1/Tm));
    KW=exp(DS_w/R - DH_w/R*(1/T-1/Tm));
    kS=km*exp(-Ei/R*(1/T-1/Tm));

    PH2=d(:,2); PCO2=d(:,3); PP=d(:,6); PH2O=d(:,7);
    drv=PP.*PH2O-(PCO2.^6.*PH2.^14)./Keq;

    switch modelName
        case 'SS_both_diss'
            den=(1+2*(KP.*PP).^0.5+2*(KW.*PH2O).^0.5).^2;
            r=kS*(KP.*KW).^0.5.*drv./den;

        case 'SS_P_diss'
            den=(1+2*(KP.*PP).^0.5+KW.*PH2O).^2;
            r=kS*(KP).^0.5.*KW.*((PP).^0.5.*PH2O)./den;

        case 'SS_W_diss'
            den=(1+KP.*PP+2*(KW.*PH2O).^0.5).^2;
            r=kS*KP.*(KW).^0.5.*(PP.*PH2O.^0.5)./den;

        case 'SS_nondiss'
            den=(1+KP.*PP+KW.*PH2O).^2;
            r=kS*KP.*KW.*drv./den;

        case 'DS_both_diss'
            den=(1+2*(KP.*PP).^0.5).*(1+2*(KW.*PH2O).^0.5);
            r=kS*(KP.*KW).^0.5.*drv./den;

        case 'DS_P_diss'
            den=(1+2*(KP.*PP).^0.5).*(1+KW.*PH2O);
            r=kS*(KP).^0.5.*KW.*((PP).^0.5.*PH2O)./den;

        case 'DS_W_diss'
            den=(1+KP.*PP).*(1+2*(KW.*PH2O).^0.5);
            r=kS*KP.*(KW).^0.5.*(PP.*PH2O.^0.5)./den;

        case 'DS_nondiss'
            den=(1+KP.*PP).*(1+KW.*PH2O);
            r=kS*KP.*KW.*drv./den;
    end
    r_pred=[r_pred;r];
end
end
