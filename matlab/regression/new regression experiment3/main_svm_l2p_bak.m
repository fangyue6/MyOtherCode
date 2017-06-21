clc;
clear;
% addpath(genpath('E:\MATLAB\R2014a\toolbox\libsvm-3.20\matlab'));


% load('EDM.mat');
% load('oes97.mat');
load('D:\fangyue\algorithm\regression\dataset\ATP7d');%加载数据集
% fea = data;
% gnd = label;
% [m n] = size(fea);
% p = randperm(m,1000);
% fea=fea(p,:);
% [m1,n1] = size(gnd);

% fea = data;%%%%%%%%%%%%%%%%%%%%%
% gnd = label;%%%%%%%%%%%%%%%%%%
fea = NormalizeFea(fea,0);
gnd = NormalizeFea(gnd,0);
% fea = stereo;
% gnd = stereolabel;
% 
[n, d] = size(fea);
label = size(gnd,2);

%% LY objective function
       para.alpha = 10^-3;%调节参数（10^-5 -- 10^5）
       para.beta = 10^-3;%调节参数（10^-5 -- 10^5）
       para.r = min(n,label)*1;
       para.p = 1.5;%调节参数  0-2
% %%  LRRR_lambda/
%       para.LRRR_lambda=1000;  
%       
% %%  SLRR_lambda
%       para.SLRR_lambda=1000; 
%       
% %%  our parameter setting
%       para.lambda=[10^2];
%       para.p =[0.5];
%       para.r=[2];
% 
% %% RFS
%       para.RFS_lambda = [1000];
%       
% %% RSR
%       para.RSR_lambda = [1000];
%       
% %% smart parameter    
%       para.smart_lambda1=1000;
%       para.smart_lambda2=1000;   
%       
% %% LSG21 parameter
%        para.LSG21_lambda=1000;
%      
% %% CSFS      
%        para.CSFSalpha=1000;
% 
% %% SFUS
%         SFUSalpha=1000;
%         SFUSbeta=1000;
%% start regression
lambda = 10;
for    j=1:length(lambda)
             ind = crossvalind('Kfold',n,10);

    for i = 1:10
        test = (ind==i);
        train = ~test;
        X_tr = fea(train,:);
        X_te = fea(test,:);
        Y_tr = gnd(train,:);
        Y_te = gnd(test,:);
        Ttest(j,i) = {ind==i};


 %% luoyan objective function
%         [AB, A, B, obj] = GL_2p_FS(X_tr',Y_tr,para.alpha,para.beta,para.r,para.p);
        [AB, A, B, obj] = GL_21_2p_FS(X_tr',Y_tr,para.alpha,para.beta,para.r,para.p);
        
         normLY = sqrt(sum(AB.*AB,2));
        [LS_LDA_Weight, LY_sorted_features] = sort(-normLY);
        percent =1;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Num_SelectFeaLY = floor(percent*d);       
        LY_SelectFeaIdx{i} = LY_sorted_features(1:Num_SelectFeaLY,:);
        LYXTrain = X_tr(:,LY_SelectFeaIdx{i});
        LYXTest = X_te(:,LY_SelectFeaIdx{i});
        %%   LRLR   LeastSquareLDA

%         [B] = LeastSquareLDA(X_tr,Y_tr, para.r);       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%   LRLR_r
%         
%         normB = sqrt(sum(B.*B,2));
%         [LS_LDA_Weight, LS_LDA_sorted_features] = sort(-normB);
%         percent = 0.9;
%         Num_SelectFea1 = floor(percent*d);       
%         LS_LDA_SelectFeaIdx{i} = LS_LDA_sorted_features(1:Num_SelectFea1,:);
%         LS_LDAXTrain = X_tr(:,LS_LDA_SelectFeaIdx{i});
%         LS_LDAXTest = X_te(:,LS_LDA_SelectFeaIdx{i});
        
 %%  LRRR    LeastSquareLDAR2    
        
%         [B] = LeastSquareLDAR2(X_tr,Y_tr, para.r, para.LRRR_lambda);  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  LRRR_lambda  r
%         normB = sqrt(sum(B.*B,2));
%         [LDAR2_Weight, LDAR2_sorted_features] = sort(-normB);
%         percent = 0.9;
%         Num_SelectFea2 = floor(percent*d);       
%         LDAR2_SelectFeaIdx{i} = LDAR2_sorted_features(1:Num_SelectFea2,:);
%         LDAR2XTrain = X_tr(:,LDAR2_SelectFeaIdx{i});
%         LDAR2XTest = X_te(:,LDAR2_SelectFeaIdx{i});

%%  SLRR      LeastSquareLDAR21

%         [B] = LeastSquareLDAR21(X_tr,Y_tr, para.SLRR_lambda, para.r);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    SLRR_lambda     r
%         normB = sqrt(sum(B.*B,2));
%         [LDAR21_Weight, LDAR21_sorted_features] = sort(-normB);
%         percent = 0.9;
%         Num_SelectFea3 = floor(percent*d);       
%         LDAR21_SelectFeaIdx{i} = LDAR21_sorted_features(1:Num_SelectFea3,:);
%         LDAR21XTrain = X_tr(:,LDAR21_SelectFeaIdx{i});
%         LDAR21XTest = X_te(:,LDAR21_SelectFeaIdx{i});
%          
 %%  our  PPDR: PP Dimensionality reduction  
%                 
%         [outA, outB, W, outNumIter, outObj] = LeastSquareLDAR2p(X_tr, Y_tr, para.r, para.p);
%         normW = sqrt(sum(W.*W,2));
%         [PPDR_Weight, PPDR_sorted_features] = sort(-normW);
%         percent = 0.9;
%         Num_SelectFea4 = floor(percent*d);       
%         PPDR_SelectFeaIdx{i} = PPDR_sorted_features(1:Num_SelectFea4,:);
%         PPDRXTrain = X_tr(:,PPDR_SelectFeaIdx{i});
%         PPDRXTest = X_te(:,PPDR_SelectFeaIdx{i});
%         

%%  RFS  2010
%         W_RFS = L21R21_inv(X_tr, Y_tr, para.RFS_lambda);
%         normW_RFS = sqrt(sum(W_RFS.*W_RFS,2));
%         [W_RFS_Weight, RFS_sorted_features] = sort(-normW_RFS);
%         percent = 0.9;
%         Num_SelectFea5 = floor(percent*d);       
%         W_RFS_SelectFeaIdx{i} = RFS_sorted_features(1:Num_SelectFea5,:);
%         RFSXTrain = X_tr(:,W_RFS_SelectFeaIdx{i});
%         RFSXTest = X_te(:,W_RFS_SelectFeaIdx{i});
        
        
%%  RSR  2015

%         W_RSR = L21R21(X_tr, Y_tr, para.RSR_lambda);
%         normW_RSR = sqrt(sum(W_RSR.*W_RSR,2));
%         [W_RSR_Weight, RSR_sorted_features] = sort(-normW_RSR);
%         percent = 0.9;
%         Num_SelectFea6 = floor(percent*d);       
%         W_RSR_SelectFeaIdx{i} = RSR_sorted_features(1:Num_SelectFea6,:);
%         RSRXTrain = X_tr(:,W_RSR_SelectFeaIdx{i});
%         RSRXTest = X_te(:,W_RSR_SelectFeaIdx{i});
%         
        
%%  smart  2011
% 
%         [W_smart,obj_smart{j,i}] = L2R1R21(X_tr,Y_tr,para.smart_lambda1,para.smart_lambda2);%%%%%%%%%%%%  smart_lambda1,  smart_lambda2
%     
%         normW_smart = sqrt(sum(W_smart.*W_smart,2));
%         [W_smart_Weight, smart_sorted_features] = sort(-normW_smart);
%         percent = 0.9;
%         Num_SelectFea7 = floor(percent*d);       
%         W_smart_SelectFeaIdx{i} = smart_sorted_features(1:Num_SelectFea7,:);
%         smartXTrain = X_tr(:,W_smart_SelectFeaIdx{i});
%         smartXTest = X_te(:,W_smart_SelectFeaIdx{i});
 
%% LSG21  2014
%         
%         [W_LSG21 ] = L2G21_new(X_tr,Y_tr, para.LSG21_lambda); %%%%%%%%%%%%%   LSG21_lambda 
%         
%         normW_LSG21 = sqrt(sum(W_LSG21.*W_LSG21,2));
%         [W_LSG21_Weight, W_LSG21_sorted_features] = sort(-normW_LSG21);
%         percent = 0.9;
%         Num_SelectFea8 = floor(percent*d);       
%         W_LSG21_SelectFeaIdx{i} = W_LSG21_sorted_features(1:Num_SelectFea8,:);
%         LSG21XTrain = X_tr(:,W_LSG21_SelectFeaIdx{i});
%         LSG21XTest = X_te(:,W_LSG21_SelectFeaIdx{i});        
        

    %% CSFS     2014-semi_supervised
%     [ W_CSFS] = CSFS_learn(X_tr,Y_tr,para.CSFSalpha);
%         normW_CSFS = sqrt(sum(W_CSFS.*W_CSFS,2));
%         [W_CSFS_Weight, W_CSFS_sorted_features] = sort(-normW_CSFS);
%         percent = 0.9;
%         Num_SelectFea9 = floor(percent*d);       
%         W_CSFS_SelectFeaIdx{i} = W_CSFS_sorted_features(1:Num_SelectFea9,:);
%         CSFSXTrain = X_tr(:,W_CSFS_SelectFeaIdx{i});
%         CSFSXTest = X_te(:,W_CSFS_SelectFeaIdx{i});

    %% SFUS  2012
%     SFUSalpha=para.SFUSalpha(k);
%     SFUSbeta=para.SFUSbeta(j);
%     W_SFUS = subspace_J21(X_tr,Y_tr,SFUSalpha,SFUSbeta);
%     
% %     W_SFUS = subspace_J21(X_tr,Y_tr,para.SFUSalpha,para.SFUSbeta);
%         normW_SFUS = sqrt(sum(W_SFUS.*W_SFUS,2));
%         [W_SFUS_Weight, W_SFUS_sorted_features] = sort(-normW_SFUS);
%         percent = 0.9;
%         Num_SelectFea10 = floor(percent*d);       
%         W_SFUS_SelectFeaIdx{i} = W_SFUS_sorted_features(1:Num_SelectFea10,:);
%         SFUSXTrain = X_tr(:,W_SFUS_SelectFeaIdx{i});
%         SFUSXTest = X_te(:,W_SFUS_SelectFeaIdx{i});


%% SVR:Support Vector Regression

  [LY.aCC(j,i),LY.aRMSE(j,i),LY.maxind(j,i)] =DD_SVR1(Y_tr,Y_te,LYXTrain,LYXTest);

%         [PPDR.aCC(j,i),PPDR.aRMSE(j,i),PCA1.aCC(j,i),PCA1.aRMSE(j,i),smart.aCC(j,i),smart.aRMSE(j,i),CSFS.aCC(j,i),CSFS.aRMSE(j,i),LSG21.aCC(j,i),LSG21.aRMSE(j,i),SFUS.aCC(j,i),SFUS.aRMSE(j,i),MSFS.aCC(j,i),MSFS.aRMSE(j,i),maxind(j,i)] =... 
%         DD_SVR3(PPDRXTrain,PPDRXTest,PCAXTrain,PCAXTest,Y_tr,Y_te,smartXTrain,smartXTest,CSFSXTrain,CSFSXTest,LSG21XTrain,LSG21XTest,SFUSXTrain,SFUSXTest,MSFSXTrain,MSFSXTest);

%         [PPDR.aCC(j,i),PPDR.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,PPDRXTrain,PPDRXTest);

%           [LRLR.aCC(j,i),LRLR.aRMSE(j,i),LRLR.maxind(j,i)] =DD_SVR1(Y_tr,Y_te,LS_LDAXTrain,LS_LDAXTest);        
%          [LRRR.aCC(j,i),LRRR.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,LDAR2XTrain,LDAR2XTest);             
%          [SLRR.aCC(j,i),SLRR.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,LDAR21XTrain,LDAR21XTest);
%          [PPDR.aCC(j,i),PPDR.aRMSE(j,i),PPDR.maxind(j,i)] =DD_SVR1(Y_tr,Y_te,PPDRXTrain,PPDRXTest);
%          [RFS.aCC(j,i),RFS.aRMSE(j,i),RFS.maxind(j,i)] =DD_SVR1(Y_tr,Y_te,RFSXTrain,RFSXTest);
%          [RSR.aCC(j,i),PPDR.aRMSE(j,i),RSR.maxind(j,i)] =DD_SVR1(Y_tr,Y_te,RSRXTrain,RSRXTest);
%           
%          [smart.aCC(j,i),smart.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,smartXTrain,smartXTest);
%          [LSG21.aCC(j,i),LSG21.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,LSG21XTrain,LSG21XTest);
% 
%          [CSFS.aCC(j,i),CSFS.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,CSFSXTrain,CSFSXTest);
%          [SFUS.aCC(j,i),SFUS.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,SFUSXTrain,SFUSXTest);
%           [NFS.aCC(j,i),NFS.aRMSE(j,i),maxind(j,i)] =DD_SVR1(Y_tr,Y_te,X_tr,X_te);
    end 
end 
meanaCC = mean(LY.aCC);
meanaRMSE = mean(LY.aRMSE);
save(['LY_ATP7d_-3_-3_1_100%Feature_result.mat'],'LY','Ttest','para','meanaCC','meanaRMSE');     

%          save(['LRLR_WQ_result'],'LRLR','maxind','Ttest','para');        
%          save(['LRRR_result'],'LRRR','maxind','Ttest','para');
%          save(['SLRR_result'],'SLRR','maxind','Ttest','para');
%          save(['PPDR_tripot20_result'],'PPDR','Ttest','para');
%          save(['RSR_SF2_R3_result'],'RSR','Ttest','para');
%          save(['RFS_SF2_R3_result'],'RFS','Ttest','para');
%          save(['smart_result'],'smart','maxind','Ttest','para'); 
%          save(['LSG21_result'],'LSG21','maxind','Ttest','para');
%         
%          save(['CSFS_result'],'CSFS','maxind','Ttest','para'); 
%          save(['SFUS_result'],'SFUS','maxind','Ttest','para');
%          save(['NFS_result'],'NFS','maxind','Ttest','para');     
%  
   
       