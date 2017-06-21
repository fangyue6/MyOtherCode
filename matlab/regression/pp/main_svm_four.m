clc;clear;
addpath(genpath('F:\matlab\toolbox\libsvm-3.20\matlab'));
addpath(genpath('F:\matlab\toolbox\libsvm-3.20'));
% load('EDM.mat');
% load('WQ.mat');
% load('ATP1d.mat');
% load('OES10.mat');
% load('RF1_train.mat');
load('ATP7d.mat');

fea = NormalizeFea(fea,0);
gnd = NormalizeFea(gnd,0);

[n, d] = size(fea);
label = size(gnd,2);
ind = crossvalind('Kfold',n,10);

%%  our parameter setting
    para.alpha = [1e+1 1e+2 1e+3 1e+4 1e+5 1e+6 1e+7 ];    %a=10^3~10^8
    para.p =0.5 ;
    para.r = min([label d]);      % r<=min[class,fea]      
%% smart parameter    
%     para.smartlambda1=1000;
%     para.smartlambda2=1000;    
%% CSFS parameter
%     para.CSFSalpha=1000;
 %%  MSFS parameter
    para.MSFSlambda1=1000;
    para.MSFSlambda2=1000;
    para.MSFSlambda3=1000;
 %% LSG21 parameter
    para.LSG21_r=1000;
 %% SFUS parameter
%     para.SFUSalpha=1000;
%     para.SFUSbeta=1000;
%     
%% start regression
for j=1:length(para.alpha)
for i = 1:10
    test = (ind==i);
    train = ~test;
    X_tr = fea(train,:);
    X_te = fea(test,:);
    Y_tr = gnd(train,:);
    Y_te = gnd(test,:);
    
    Ttest(j,i) = {ind==i};
    
%% PPDR: PP Dimensionality reduction   our
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

% %     normB = sqrt(sum(B.*B,2));
% %     normB(normB<=0.8*mean(normB)) = 0;
% %     SelectFeaIdx{i} = find(normB~=0);
% %     NewXTrain = X_tr(:,SelectFeaIdx{i});
% %     NewXTest = X_te(:,SelectFeaIdx{i});

%% PCA 
%     options.ReducedDim= Num_SelectFea;
%     [eigvector,eigvalue] = PCA(fea,options);
%     PCA_Fea = fea*eigvector;
%     PCAXTrain = PCA_Fea(train,:);
%     PCAXTest = PCA_Fea(test,:);
    
%%  smart  2011
%     [W_smart,obj_smart{j,i}] = L2R1R21(X_tr,Y_tr,para.smartlambda1,para.smartlambda2);
% 
%     % 2. feature selection;
%     normW_smart = sqrt(sum(W_smart.*W_smart,2));
%     [W_smart_Weight, smart_sorted_features] = sort(-normW_smart);
%     percent = 0.9;
%     Num_SelectFea = floor(percent*d);       
%     W_smart_SelectFeaIdx{i} = smart_sorted_features(1:Num_SelectFea,:);
%     smartXTrain = X_tr(:,W_smart_SelectFeaIdx{i});
%     smartXTest = X_te(:,W_smart_SelectFeaIdx{i});
    
%% MSFS   2014
   [W_MSFS]= LF3L21(X_tr,Y_tr,para.MSFSlambda1,para.MSFSlambda2,para.MSFSlambda3);
    normW_MSFS = sqrt(sum(W_MSFS.*W_MSFS,2));
    [W_MSFS_Weight, W_MSFS_sorted_features] = sort(-normW_MSFS);
    percent = 0.9;
    Num_SelectFea = floor(percent*d);       
    W_MSFS_SelectFeaIdx{i} = W_MSFS_sorted_features(1:Num_SelectFea,:);
    MSFSXTrain = X_tr(:,W_MSFS_SelectFeaIdx{i});
    MSFSXTest = X_te(:,W_MSFS_SelectFeaIdx{i});
%% CSFS     2014-semi_supervised
% [ W_CSFS] = CSFS_learn(X_tr,Y_tr,para.CSFSalpha);
%     normW_CSFS = sqrt(sum(W_CSFS.*W_CSFS,2));
%     [W_CSFS_Weight, W_CSFS_sorted_features] = sort(-normW_CSFS);
%     percent = 0.9;
%     Num_SelectFea = floor(percent*d);       
%     W_CSFS_SelectFeaIdx{i} = W_CSFS_sorted_features(1:Num_SelectFea,:);
%     CSFSXTrain = X_tr(:,W_CSFS_SelectFeaIdx{i});
%     CSFSXTest = X_te(:,W_CSFS_SelectFeaIdx{i});

%% LSG21  2014
[W_LSG21 ] = L2G21_new(X_tr,Y_tr,para.LSG21_r);
    normW_LSG21 = sqrt(sum(W_LSG21.*W_LSG21,2));
    [W_LSG21_Weight, W_LSG21_sorted_features] = sort(-normW_LSG21);
    percent = 0.9;
    Num_SelectFea = floor(percent*d);       
    W_LSG21_SelectFeaIdx{i} = W_LSG21_sorted_features(1:Num_SelectFea,:);
    LSG21XTrain = X_tr(:,W_LSG21_SelectFeaIdx{i});
    LSG21XTest = X_te(:,W_LSG21_SelectFeaIdx{i});
    
%% SFUS  2012
% W_SFUS = subspace_J21(X_tr,Y_tr,para.SFUSalpha,para.SFUSbeta);
%     normW_SFUS = sqrt(sum(W_SFUS.*W_SFUS,2));
%     [W_SFUS_Weight, W_SFUS_sorted_features] = sort(-normW_SFUS);
%     percent = 0.9;
%     Num_SelectFea = floor(percent*d);       
%     W_SFUS_SelectFeaIdx{i} = W_SFUS_sorted_features(1:Num_SelectFea,:);
%     SFUSXTrain = X_tr(:,W_SFUS_SelectFeaIdx{i});
%     SFUSXTest = X_te(:,W_SFUS_SelectFeaIdx{i});

%% SVR:Support Vector Regression
%     [PPDR.aCC(j,i),PPDR.aRMSE(j,i),PCA1.aCC(j,i),PCA1.aRMSE(j,i),smart.aCC(j,i),smart.aRMSE(j,i),CSFS.aCC(j,i),CSFS.aRMSE(j,i),LSG21.aCC(j,i),LSG21.aRMSE(j,i),SFUS.aCC(j,i),SFUS.aRMSE(j,i),MSFS.aCC(j,i),MSFS.aRMSE(j,i),maxind(j,i)] =... 
%     DD_SVR3(PPDRXTrain,PPDRXTest,PCAXTrain,PCAXTest,Y_tr,Y_te,smartXTrain,smartXTest,CSFSXTrain,CSFSXTest,LSG21XTrain,LSG21XTest,SFUSXTrain,SFUSXTest,MSFSXTrain,MSFSXTest);

 [PPDR.aCC(j,i),PPDR.aRMSE(j,i),LSG21.aCC(j,i),LSG21.aRMSE(j,i),MSFS.aCC(j,i),MSFS.aRMSE(j,i),maxind(j,i)] =DD_SVR3(PPDRXTrain,PPDRXTest,Y_tr,Y_te,LSG21XTrain,LSG21XTest,MSFSXTrain,MSFSXTest);
end 
end
% save(['ATP1d_result'],'PPDR','PCA1','smart','CSFS','LSG21','SFUS','MSFS','maxind','Ttest','para','obj');
save(['ATP7d_result'],'PPDR','LSG21','MSFS','maxind','Ttest','para','obj');
a_ppdr=[min(PPDR.aCC),max(PPDR.aCC),mean(PPDR.aCC);min(PPDR.aRMSE),max(PPDR.aRMSE),mean(PPDR.aRMSE)]
% a_pca1=[min(PCA1.aCC),max(PCA1.aCC),mean(PCA1.aCC);min(PCA1.aRMSE),max(PCA1.aRMSE),mean(PCA1.aRMSE)]
% a_smart=[min(smart.aCC),max(smart.aCC),mean(smart.aCC);min(smart.aRMSE),max(smart.aRMSE),mean(smart.aRMSE)]
% a_csfs=[min(CSFS.aCC),max(CSFS.aCC),mean(CSFS.aCC);min(CSFS.aRMSE),max(CSFS.aRMSE),mean(CSFS.aRMSE)]
a_lsg21=[min(LSG21.aCC),max(LSG21.aCC),mean(LSG21.aCC);min(LSG21.aRMSE),max(LSG21.aRMSE),mean(LSG21.aRMSE)]
% a_sfus=[min(SFUS.aCC),max(SFUS.aCC),mean(SFUS.aCC);min(SFUS.aRMSE),max(SFUS.aRMSE),mean(SFUS.aRMSE)]
a_msfs=[min(MSFS.aCC),max(MSFS.aCC),mean(MSFS.aCC);min(MSFS.aRMSE),max(MSFS.aRMSE),mean(MSFS.aRMSE)]
%%