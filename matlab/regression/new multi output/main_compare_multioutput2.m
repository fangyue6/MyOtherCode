clc;
clear;
% addpath(genpath('E:\MATLAB\R2014a\toolbox\libsvm-3.20\matlab'));
currentFolder = pwd;
% addpath(genpath([currentFolder,'\FRFS0']));
addpath(genpath([currentFolder,'\MIMLfast']));

dataname='scm1d_test';
% load('EDM.mat');
% load('oes97.mat');
load(['datasets/',dataname,'.mat']);%加载数据集
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

%% Our objective function
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

        
%        %% result_FSSI
%         para_FSSI.alpha = 1;
%         para_FSSI.beta = 1;
%         [W,~] = FSSI(X_tr',Y_tr',para_FSSI);
%         normOur = sqrt(sum(W.*W,2));
%         [LS_LDA_Weight, result_FSSI_sorted_features] = sort(-normOur);
%         para_FSSI.percent =1;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Num_SelectFearesult_FSSI = floor(para_FSSI.percent*d);       
%         result_FSSI_SelectFeaIdx{i} = result_FSSI_sorted_features(1:Num_SelectFearesult_FSSI,:);
%         result_FSSIXTrain = X_tr(:,result_FSSI_SelectFeaIdx{i});
%         result_FSSIXTest = X_te(:,result_FSSI_SelectFeaIdx{i});
%         [result_FSSI.aCC(j,i),result_FSSI.aRE(j,i),result_FSSI.MSE(j,i),result_FSSI.aRMSE(j,i),result_FSSI.aRRMSE(j,i),result_FSSI.MAE(j,i)] =...
%         DD_SVR_By_FY(Y_tr,Y_te,result_FSSIXTrain,result_FSSIXTest);%aCC, aRE, MSE, aRMSE, aRRMSE, MAE
% 
%       
% 
%        %%  RFS  2010
%         para_RFS.lambda=1;
%         W_RFS = RFS(X_tr, Y_tr, para_RFS.lambda);
%         normW_RFS = sqrt(sum(W_RFS.*W_RFS,2));
%         [W_RFS_Weight, RFS_sorted_features] = sort(-normW_RFS);
%         para_RFS.percent = 0.9;
%         Num_SelectFea_RFS = floor(para_RFS.percent*d);       
%         W_RFS_SelectFeaIdx{i} = RFS_sorted_features(1:Num_SelectFea_RFS,:);
%         RFSXTrain = X_tr(:,W_RFS_SelectFeaIdx{i});
%         RFSXTest = X_te(:,W_RFS_SelectFeaIdx{i});
%         [result_RFS.aCC(j,i),result_RFS.aRE(j,i),result_RFS.MSE(j,i),result_RFS.aRMSE(j,i),result_RFS.aRRMSE(j,i),result_RFS.MAE(j,i)] ...
%         =DD_SVR_By_FY(Y_tr,Y_te,RFSXTrain,RFSXTest);%aCC, aRE, MSE, aRMSE, aRRMSE, MAE
%     
%  
%        %% CSFS     2014-semi_supervised
%         para_CSFS.alpha = 1;
%         [ W_CSFS] = CSFS_learn(X_tr,Y_tr,para_CSFS.alpha);
%         normW_CSFS = sqrt(sum(W_CSFS.*W_CSFS,2));
%         [W_CSFS_Weight, W_CSFS_sorted_features] = sort(-normW_CSFS);
%         para_CSFS.percent = 0.9;
%         Num_SelectFea_CSFS = floor(para_CSFS.percent*d);       
%         W_CSFS_SelectFeaIdx{i} = W_CSFS_sorted_features(1:Num_SelectFea_CSFS,:);
%         CSFSXTrain = X_tr(:,W_CSFS_SelectFeaIdx{i});
%         CSFSXTest = X_te(:,W_CSFS_SelectFeaIdx{i});
%         [result_CSFS.aCC(j,i),result_CSFS.aRE(j,i),result_CSFS.MSE(j,i),result_CSFS.aRMSE(j,i),result_CSFS.aRRMSE(j,i),result_CSFS.MAE(j,i)] ...
%         =DD_SVR_By_FY(Y_tr,Y_te,CSFSXTrain,CSFSXTest);%aCC, aRE, MSE, aRMSE, aRRMSE, MAE
    
    
       %% MIMLfast
       [nX_tr,~] = size(X_tr);
        for im=1:nX_tr
            MIML_train_data{im,1} = X_tr(im,:);
        end
       [nY_tr,~] = size(Y_tr);
        for ij=1:nY_tr
            max_Y_tr = max(Y_tr(ij,:));
            min_Y_tr = min(Y_tr(ij,:));
            big = (Y_tr(ij,:)>=(max_Y_tr + min_Y_tr)/2);
%             small = 1-(Y_tr(ij,:)>=(max_Y_tr + min_Y_tr)/2);
            Y_tr(ij,big)=1;
            Y_tr(ij,Y_tr(ij,:)~=1)=-1;
%             Y_tr(ij,small)=double(-1);
        end
%         MIML_train_data{1} = X_tr;
%          max_Y_tr = max(max(Y_tr));
%          min_Y_tr = min(min(Y_tr));
%          Y_tr(Y_tr>=(max_Y_tr+min_Y_tr)/2)=1;
%          Y_tr(Y_tr<(max_Y_tr+min_Y_tr)/2)=-1;
        [MIML_W]=MIMLfast1(MIML_train_data,Y_tr);

        MIML_W(MIML_W>0)=1;
        MIML_W(MIML_W<=0)=0;
        
        normW_MIML = sqrt(sum(MIML_W.*MIML_W,2));
        [W_MIML_Weight, W_MIML_sorted_features] = sort(-normW_MIML);
        para_MIML.percent = 0.9;
        Num_SelectFea_MIML = floor(para_MIML.percent*d);       
        W_MIML_SelectFeaIdx{i} = W_MIML_sorted_features(1:Num_SelectFea_MIML,:);
        MIMLXTrain = X_tr(:,W_MIML_SelectFeaIdx{i});
        MIMLXTest = X_te(:,W_MIML_SelectFeaIdx{i});
        [result_MIML.aCC(j,i),result_MIML.aRE(j,i),result_MIML.MSE(j,i),result_MIML.aRMSE(j,i),result_MIML.aRRMSE(j,i),result_MIML.MAE(j,i)] ...
        =DD_SVR_By_FY(Y_tr,Y_te,MIMLXTrain,MIMLXTest);%aCC, aRE, MSE, aRMSE, aRRMSE, MAE







% %% FSASL
% classnum = length(unique(gnd));
% [W,~,~,~] = FSASL(X_tr',classnum);
% AB=W;
%         
%         
% %% FRFS0  zhupengfei:Non-convex Regularized Self-representation for Unsupervised Feature Selection
%         [W,obj] = FRFS0(X_tr,0.001,0.1);
%         AB=W;
% normOur = sqrt(sum(AB.*AB,2));
% [LS_LDA_Weight, Our_sorted_features] = sort(-normOur);
% percent =1;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Num_SelectFeaOur = floor(percent*d);       
% Our_SelectFeaIdx{i} = Our_sorted_features(1:Num_SelectFeaOur,:);
% OurXTrain = X_tr(:,Our_SelectFeaIdx{i});
% OurXTest = X_te(:,Our_SelectFeaIdx{i});
        
%% fangyue objective function
% %         [AB, A, B, obj] = GL_2p_FS(X_tr',Y_tr,para.alpha,para.beta,para.r,para.p);
%         [AB, A, B, obj] = GL_21_2p_FS(X_tr',Y_tr,para.alpha,para.beta,para.r,para.p);
%         
%         normOur = sqrt(sum(AB.*AB,2));
%         [LS_LDA_Weight, Our_sorted_features] = sort(-normOur);
%         percent =1;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Num_SelectFeaOur = floor(percent*d);       
%         Our_SelectFeaIdx{i} = Our_sorted_features(1:Num_SelectFeaOur,:);
%         OurXTrain = X_tr(:,Our_SelectFeaIdx{i});
%         OurXTest = X_te(:,Our_SelectFeaIdx{i});
        
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

%   [Our.aCC(j,i),Our.aRMSE(j,i),Our.maxind(j,i)] =DD_SVR1(Y_tr,Y_te,OurXTrain,OurXTest);
%   [Our.aCC(j,i),Our.aRE(j,i),Our.MSE(j,i),Our.aRMSE(j,i),Our.aRRMSE(j,i),Our.MAE(j,i)] =DD_SVR_By_FY(Y_tr,Y_te,OurXTrain,OurXTest);%aCC, aRE, MSE, aRMSE, aRRMSE, MAE

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
% Our.aCC(j,i),Our.aRE(j,i),Our.MSE(j,i),Our.aRMSE(j,i),Our.aRRMSE(j,i),Our.MAE(j,i)
% meanaCC = mean(Our.aCC);
% meanaRE = mean(Our.aRE);
% meanMSE = mean(Our.MSE);
% meanaRMSE = mean(Our.aRMSE);
% meanaRRMSE = mean(Our.aRRMSE);
% meanMAE = mean(Our.MAE);

% meanvalue_FSSI = meanEvaluate_by_FY(result_FSSI)
% save(['result/FSSI/',dataname,'_',num2str(meanvalue_FSSI(1)),'.mat'],'result_FSSI','para_FSSI','meanvalue_FSSI'); 
% 
% meanvalue_RFS = meanEvaluate_by_FY(result_RFS)
% save(['result/RFS/',dataname,'_',num2str(meanvalue_RFS(1)),'.mat'],'result_RFS','para_RFS','meanvalue_RFS'); 
% 
% meanvalue_CSFS =  meanEvaluate_by_FY(result_CSFS)
% save(['result/CSFS/',dataname,'_',num2str(meanvalue_CSFS(1)),'.mat'],'result_CSFS','para_CSFS','meanvalue_CSFS');  

meanvalue_MIML = meanEvaluate_by_FY(result_MIML)
save(['result/MIML/',dataname,'_',num2str(meanvalue_MIML(1)),'.mat'],'result_MIML','para_MIML','meanvalue_MIML'); 

% save([dataname,'_-3_-3_1_100%Feature_result.mat'],'result_FSSI','Ttest','para','meanaCC','meanaRMSE');     

  
%  
   
       