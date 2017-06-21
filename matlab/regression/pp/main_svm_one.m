clc;clear;
addpath(genpath('F:\matlab\toolbox\libsvm-3.20\matlab'));
addpath(genpath('F:\matlab\toolbox\libsvm-3.20'));
% load('EDM.mat');
% load('WQ.mat');
% load('ATP1d.mat');
% load('OES10.mat');
% load('RF1_train.mat');
load('ATP1d.mat');

fea = NormalizeFea(fea,0);
gnd = NormalizeFea(gnd,0);

[n, d] = size(fea);
label = size(gnd,2);
% ind = crossvalind('Kfold',n,10);

%%  our parameter setting
%  %    para.alpha = [1e+1 1e+2 1e+3 1e+4 1e+5 1e+6 1e+7 ];    %a=10^3~10^8
% %   para.alpha = [1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7];
%  para.alpha = 1e+3;
% %    para.p =[0.1:0.1:1] ;
%     para.p =0.5;
% %     para.r = min([label d]);      % r<=min[class,fea]  
%     para.r =4;
%% smart parameter    
    para.smartlambda1=[1e+1 ];
    para.smartlambda2=[ 1e+7 ];    
%% CSFS parameter
%     para.CSFSalpha=1000;
 %%  MSFS parameter
%     para.MSFSlambda1=1000;
%     para.MSFSlambda2=1000;
%     para.MSFSlambda3=1000;
 %% LSG21 parameter
%     para.LSG21_r=1000;
 %% SFUS parameter
%     para.SFUSalpha=1000;
%     para.SFUSbeta=1000;
     
%% start regression
for k=1:length(para.smartlambda1)
    for j=1:length(para.smartlambda2)
    ind = crossvalind('Kfold',n,10);
for i = 1:10
    test = (ind==i);
    train = ~test;
    X_tr = fea(train,:);
    X_te = fea(test,:);
    Y_tr = gnd(train,:);
    Y_te = gnd(test,:);
    
    Ttest(j,i) = {ind==i};
    
%% PPDR: PP Dimensionality reduction   our
%     p=para.p(j);
%     alpha=para.alpha;
%     % 1. generate regression coefficient
%     [B,A,W,obj{j,i}] = RRPS(X_tr,Y_tr,alpha,p,para.r);
% 
%     % 2. feature selection;
%     normB = sqrt(sum(B.*B,2));
%     [PPDR_Weight, PPDR_sorted_features] = sort(-normB);
%     percent = 0.9;
%     Num_SelectFea = floor(percent*d);       
%     PPDR_SelectFeaIdx{i} = PPDR_sorted_features(1:Num_SelectFea,:);
%     PPDRXTrain = X_tr(:,PPDR_SelectFeaIdx{i});
%     PPDRXTest = X_te(:,PPDR_SelectFeaIdx{i});

% % %     normB = sqrt(sum(B.*B,2));
% % %     normB(normB<=0.8*mean(normB)) = 0;
% % %     SelectFeaIdx{i} = find(normB~=0);
% % %     NewXTrain = X_tr(:,SelectFeaIdx{i});
% % %     NewXTest = X_te(:,SelectFeaIdx{i});

%% PCA 
%     options.ReducedDim= Num_SelectFea;
%     [eigvector,eigvalue] = PCA(fea,options);
%     PCA_Fea = fea*eigvector;
%     PCAXTrain = PCA_Fea(train,:);
%     PCAXTest = PCA_Fea(test,:);
    
%%  smart  2011

smartlambda1=para.smartlambda1(k);
smartlambda2=para.smartlambda2(j);

    [W_smart,obj_smart{k,j,i}] = L2R1R21(X_tr,Y_tr,smartlambda1,smartlambda2);

    % 2. feature selection;
    normW_smart = sqrt(sum(W_smart.*W_smart,2));
    [W_smart_Weight, smart_sorted_features] = sort(-normW_smart);
    percent = 0.9;
    Num_SelectFea = floor(percent*d);       
    W_smart_SelectFeaIdx{i} = smart_sorted_features(1:Num_SelectFea,:);
    smartXTrain = X_tr(:,W_smart_SelectFeaIdx{i});
    smartXTest = X_te(:,W_smart_SelectFeaIdx{i});
    
%% MSFS   2014
%    [W_MSFS]= LF3L21(X_tr,Y_tr,para.MSFSlambda1,para.MSFSlambda2,para.MSFSlambda3);
%     normW_MSFS = sqrt(sum(W_MSFS.*W_MSFS,2));
%     [W_MSFS_Weight, W_MSFS_sorted_features] = sort(-normW_MSFS);
%     percent = 0.9;
%     Num_SelectFea = floor(percent*d);       
%     W_MSFS_SelectFeaIdx{i} = W_MSFS_sorted_features(1:Num_SelectFea,:);
%     MSFSXTrain = X_tr(:,W_MSFS_SelectFeaIdx{i});
%     MSFSXTest = X_te(:,W_MSFS_SelectFeaIdx{i});
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
% [W_LSG21 ] = L2G21_new(X_tr,Y_tr,para.LSG21_r);
%     normW_LSG21 = sqrt(sum(W_LSG21.*W_LSG21,2));
%     [W_LSG21_Weight, W_LSG21_sorted_features] = sort(-normW_LSG21);
%     percent = 0.9;
%     Num_SelectFea = floor(percent*d);       
%     W_LSG21_SelectFeaIdx{i} = W_LSG21_sorted_features(1:Num_SelectFea,:);
%     LSG21XTrain = X_tr(:,W_LSG21_SelectFeaIdx{i});
%     LSG21XTest = X_te(:,W_LSG21_SelectFeaIdx{i});
    
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

[smart.aCC(k,j,i),smart.aRMSE(k,j,i),maxind(k,j,i)] =DD_SVR1(Y_tr,Y_te,smartXTrain,smartXTest);
end 
    end
end
% save(['ATP1d_result'],'PPDR','PCA1','smart','CSFS','LSG21','SFUS','MSFS','maxind','Ttest','para','obj');
 save(['ATP1d_result'],'smart','maxind','Ttest','para');

