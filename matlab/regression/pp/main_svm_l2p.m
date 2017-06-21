clc;
clear;
addpath(genpath('F:\matlab\toolbox\libsvm-3.20\matlab'));
addpath(genpath('F:\matlab\toolbox\libsvm-3.20'));
addpath('C:\Users\sdjsj\Desktop\L2p\pp\mtl_feat_trace');

load('EDM.mat');
% load('oes97.mat');
% load('WQ.mat');
% load('scm1d_test.mat')
% load('OES10.mat');
% load('sf1_m.mat');
% load('ATP1d.mat');
% load('scm20_test.mat')

fea = NormalizeFea(fea,0);
gnd = NormalizeFea(gnd,0);
[n, d] = size(fea);
label = size(gnd,2);

%%  LRRR_lambda
%       para.LRRR_lambda=100;  

%%  SLRR_lambda
%       para.SLRR_lambda=0.001;

%%  our parameter setting
%       para.our_alpha=[1e+1 1e+2 1e+3 1e+4 1e+5 1e+6 1e+7 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 ];
%       para.p =[0.1 0.3 0.5 0.7 0.9];
      para.our_alpha=[0];
      para.p =[ 0.5 ];
      para.r=[16];
      
%% smart parameter    
%       para.smart_lambda1=10000;
%       para.smart_lambda2=1000;   
%       
%% LSG21 parameter
%       para.LSG21_lambda=10000;
      
%% CSFS parameter
%       para.CSFS_lambda=[1e+3 ];

%% trace_norm
para.gammas=[1e+3 ];

%% start regression

for    j=1:length(para.gammas)
    gammas=para.gammas(j);
%        for pp=1:length(para.p)
             ind = crossvalind('Kfold',n,10);

    for i = 1:10
        test = (ind==i);
        train = ~test;
        X_tr = fea(train,:);
        X_te = fea(test,:);
        Y_tr = gnd(train,:);
        Y_te = gnd(test,:);
        Ttest(j,i) = {ind==i};

%%   LRLR   LeastSquareLDA

%         [B,A] = LeastSquareLDA(X_tr,Y_tr, para.r);       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%   LRLR_r
%         W=B*A;
%         normB = sqrt(sum(W.*W,2));
%         [LS_LDA_Weight, LS_LDA_sorted_features] = sort(-normB);
%         percent = 0.9;
%         Num_SelectFea = floor(percent*d);       
%         LS_LDA_SelectFeaIdx{i} = LS_LDA_sorted_features(1:Num_SelectFea,:);
%         LS_LDAXTrain = X_tr(:,LS_LDA_SelectFeaIdx{i});
%         LS_LDAXTest = X_te(:,LS_LDA_SelectFeaIdx{i});
%         
 %%  LRRR    LeastSquareLDAR2    
        
%         [B,A] = LeastSquareLDAR2(X_tr,Y_tr, para.r, para.LRRR_lambda);  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  LRRR_lambda  r
%          W=B*A;
%         normB = sqrt(sum(W.*W,2));
%         [LDAR2_Weight, LDAR2_sorted_features] = sort(-normB);
%         percent = 0.9;
%         Num_SelectFea = floor(percent*d);       
%         LDAR2_SelectFeaIdx{i} = LDAR2_sorted_features(1:Num_SelectFea,:);
%         LDAR2XTrain = X_tr(:,LDAR2_SelectFeaIdx{i});
%         LDAR2XTest = X_te(:,LDAR2_SelectFeaIdx{i});

%%  SLRR      LeastSquareLDAR21

%         [B,A] = LeastSquareLDAR21(X_tr,Y_tr, para.SLRR_lambda, para.r);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    SLRR_lambda     r
%         W=B*A;
%         normB = sqrt(sum(W.*W,2));
%         [LDAR21_Weight, LDAR21_sorted_features] = sort(-normB);
%         percent = 0.9;
%         Num_SelectFea = floor(percent*d);       
%         LDAR21_SelectFeaIdx{i} = LDAR21_sorted_features(1:Num_SelectFea,:);
%         LDAR21XTrain = X_tr(:,LDAR21_SelectFeaIdx{i});
%         LDAR21XTest = X_te(:,LDAR21_SelectFeaIdx{i});
         
 %%  our  PPDR: PP Dimensionality reduction  
        
%         [B,A,W,obj{j,pp,i}] = RRPS(X_tr,Y_tr, para.our_alpha(j), para.p(pp), para.r);    %%%%%%%%%%%%%%%%%%%%%     our_alpha   p   r
%         normB = sqrt(sum(B.*B,2));
%         [PPDR_Weight, PPDR_sorted_features] = sort(-normB);
%         percent = 0.9;
%         Num_SelectFea = floor(percent*d);       
%         PPDR_SelectFeaIdx{i} = PPDR_sorted_features(1:Num_SelectFea,:);
%         PPDRXTrain = X_tr(:,PPDR_SelectFeaIdx{i});
%         PPDRXTest = X_te(:,PPDR_SelectFeaIdx{i});

%%  smart  2011

%         [W_smart,obj_smart{j,i}] = L2R1R21(X_tr,Y_tr,para.smart_lambda1,para.smart_lambda2);%%%%%%%%%%%%  smart_lambda1,  smart_lambda2
%     
%         normW_smart = sqrt(sum(W_smart.*W_smart,2));
%         [W_smart_Weight, smart_sorted_features] = sort(-normW_smart);
%         percent = 0.9;
%         Num_SelectFea = floor(percent*d);       
%         W_smart_SelectFeaIdx{i} = smart_sorted_features(1:Num_SelectFea,:);
%         smartXTrain = X_tr(:,W_smart_SelectFeaIdx{i});
%         smartXTest = X_te(:,W_smart_SelectFeaIdx{i});
 
%% LSG21  2014
        
%         [W_LSG21 ] = L2G21_new(X_tr,Y_tr, para.LSG21_lambda); %%%%%%%%%%%%%   LSG21_lambda 
%         
%         normW_LSG21 = sqrt(sum(W_LSG21.*W_LSG21,2));
%         [W_LSG21_Weight, W_LSG21_sorted_features] = sort(-normW_LSG21);
%         percent = 0.9;
%         Num_SelectFea = floor(percent*d);       
%         W_LSG21_SelectFeaIdx{i} = W_LSG21_sorted_features(1:Num_SelectFea,:);
%         LSG21XTrain = X_tr(:,W_LSG21_SelectFeaIdx{i});
%         LSG21XTest = X_te(:,W_LSG21_SelectFeaIdx{i}); 

%% CSFS     2014-semi_supervised
%     [ W_CSFS] = CSFS_learn(X_tr,Y_tr,CSFS_lambda);
%         normW_CSFS = sqrt(sum(W_CSFS.*W_CSFS,2));
%         [W_CSFS_Weight, W_CSFS_sorted_features] = sort(-normW_CSFS);
%         percent = 0.9;
%         Num_SelectFea = floor(percent*d);       
%         W_CSFS_SelectFeaIdx{i} = W_CSFS_sorted_features(1:Num_SelectFea,:);
%         CSFSXTrain = X_tr(:,W_CSFS_SelectFeaIdx{i});
%         CSFSXTest = X_te(:,W_CSFS_SelectFeaIdx{i});   

     %% MSFS   2014
       [W_MSFS]= LF3L21(X_tr,Y_tr,para.MSFSlambda1,para.MSFSlambda2,para.MSFSlambda3);
        normW_MSFS = sqrt(sum(W_MSFS.*W_MSFS,2));
        [W_MSFS_Weight, W_MSFS_sorted_features] = sort(-normW_MSFS);
        percent = 0.9;
        Num_SelectFea = floor(percent*d);       
        W_MSFS_SelectFeaIdx{i} = W_MSFS_sorted_features(1:Num_SelectFea,:);
        MSFSXTrain = X_tr(:,W_MSFS_SelectFeaIdx{i});
        MSFSXTest = X_te(:,W_MSFS_SelectFeaIdx{i});

    %% SFUS  2012
%     SFUSalpha=para.SFUSalpha(k);
%     SFUSbeta=para.SFUSbeta(j);
%     W_SFUS = subspace_J21(X_tr,Y_tr,SFUSalpha,SFUSbeta);
%     
% %     W_SFUS = subspace_J21(X_tr,Y_tr,para.SFUSalpha,para.SFUSbeta);
%         normW_SFUS = sqrt(sum(W_SFUS.*W_SFUS,2));
%         [W_SFUS_Weight, W_SFUS_sorted_features] = sort(-normW_SFUS);
%         percent = 0.9;
%         Num_SelectFea = floor(percent*d);       
%         W_SFUS_SelectFeaIdx{i} = W_SFUS_sorted_features(1:Num_SelectFea,:);
%         SFUSXTrain = X_tr(:,W_SFUS_SelectFeaIdx{i});
%         SFUSXTest = X_te(:,W_SFUS_SelectFeaIdx{i});

%% trace norm
Dini=zeros(d,d);
for idi=1:1:d
    for jdi=1:1:d
Dini(idi,jdi) =1/d;
    end
end
iterations=20;
[mm,nn]=size(gnd);
task_indexes=zeros(nn,1);

for tti=1:1:nn
task_indexes(tti,1)=tti;
end

method_str='feat';
[theW] = run_code_example(gammas,X_tr,Y_tr,task_indexes,Dini,iterations,method_str,0);

        normtheW = sqrt(sum(theW.*theW,2));
        [theW_Weight, theW_sorted_features] = sort(-normtheW);
        percent = 0.9;
        Num_SelectFea = floor(percent*d);       
        theW_SelectFeaIdx{i} = theW_sorted_features(1:Num_SelectFea,:);
        theWXTrain = X_tr(:,theW_SelectFeaIdx{i});
        theWXTest = X_te(:,theW_SelectFeaIdx{i});

%% SVR:Support Vector Regression

%          [LRLR.aCC(j,i),LRLR.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,LS_LDAXTrain,LS_LDAXTest);        
%          [LRRR.aCC(j,i),LRRR.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,LDAR2XTrain,LDAR2XTest);
%          [SLRR.aCC(j,i),SLRR.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,LDAR21XTrain,LDAR21XTest);
%          [PPDR.aCC(j,pp,i),PPDR.aRMSE(j,pp,i),maxind(j,pp,i)] =DD_SVR1(Y_tr,Y_te,PPDRXTrain,PPDRXTest);
%          [smart.aCC(j,i),smart.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,smartXTrain,smartXTest);
%          [LSG21.aCC(j,i),LSG21.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,LSG21XTrain,LSG21XTest);
%            [CSFS.aCC(j,i),CSFS.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,CSFSXTrain,CSFSXTest);
 [trace_norm.aCC(j,i),trace_norm.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,theWXTrain,theWXTest);

    end 
%        end
end   
%         save(['LRLR_result'],'LRLR','maxind','Ttest','para');        
%         save(['LRRR_result'],'LRRR','maxind','Ttest','para');
%         save(['SLRR_result'],'SLRR','maxind','Ttest','para');
%         save(['PPDR_result'],'PPDR','maxind','Ttest','para');
%         save(['smart_result'],'smart','maxind','Ttest','para'); 
%         save(['LSG21_result'],'LSG21','maxind','Ttest','para'); 
%           save(['CSFS_result'],'CSFS','maxind','Ttest','para'); 
save(['trace_norm_result'],'trace_norm','maxind','Ttest','para'); 
   
       