clc;clear;
addpath(genpath('F:\matlab\toolbox\libsvm-3.20\matlab'));
addpath(genpath('F:\matlab\toolbox\libsvm-3.20'));
% load('dataset\EDM.mat');
load('WQ.mat');
% load('ATP1d.mat');

fea = NormalizeFea(fea,0);
gnd = NormalizeFea(gnd,0);
[n, d] = size(fea);
label = size(gnd,2);

%%  parameter setting
    para.alpha = [1e+1 1e+2 1e+3 1e+4 1e+5 1e+6 1e+7];    %a=10^3~10^8
    para.p =0.5 ;
    para.r = min([label d]);      % r<=min[class,fea]    
    
    ind = crossvalind('Kfold',n,10);
for j=1:length(para.alpha)
   
for i = 1:10
    test = (ind==i);
    train = ~test;
    X_tr = fea(train,:);
    X_te = fea(test,:);
    Y_tr = gnd(train,:);
    Y_te = gnd(test,:);
    
    Ttest(j,i) = {ind==i};
    
%% PPDR: PP Dimensionality reduction
    p=para.p;
    alpha=para.alpha(j);
    % 1. generate regression coefficient
    [B,A,W,obj{j,i}] = RRPS(X_tr,Y_tr,alpha,p,para.r);

    % 2. feature selection;
    normB = sqrt(sum(B.*B,2));
    [PPDR_Weight, PPDR_sorted_features] = sort(-normB);
    percent = 0.9;
    Num_SelectFea = floor(percent*d);       
    PPDR_SelectFeaIdx{i} = PPDR_sorted_features(1:Num_SelectFea,:);
    PPDRXTrain = X_tr(:,PPDR_SelectFeaIdx{i});
    PPDRXTest = X_te(:,PPDR_SelectFeaIdx{i});
    
%% SVR:Support Vector Regression
    [PPDR.aCC(j,i),PPDR.aRMSE(j,i),maxind(j,i)] = DD_SVR1(PPDRXTrain,PPDRXTest,Y_tr,Y_te);

end 
end
save(['ATP1d_result'],'PPDR','maxind','Ttest','para','obj');

%%
