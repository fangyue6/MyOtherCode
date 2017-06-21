clc;clear;
% addpath(genpath('F:\matlab\toolbox\libsvm-3.20\matlab'));
% addpath(genpath('F:\matlab\toolbox\libsvm-3.20'));

dataname='sf2_m';

% CSFS LSG21 SLRR RSR MSFS
compareName='CSFS';
load(['D:\fangyue\algorithm\regression\dataset\',dataname,'.mat']);


fea = data;
gnd = label;
fea = NormalizeFea(fea,0);
gnd = NormalizeFea(gnd,0);
[n, d] = size(fea);
label = size(gnd,2);

% %% parameter
%        alpha = 3;
%        beta = -2;
%        p = 1.2;
%        para.alpha = 10^alpha;%调节参数（10^-5 -- 10^5）
%        para.beta = 10^beta;%调节参数（10^-5 -- 10^5）
%        para.r = min(n,label)*1;
%        para.p = p;%调节参数  0-2

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

        switch compareName
            case 'CSFS'
              %% CSFS     2014-semi_supervised
                % CSFS parameter
                para.CSFS_lambda=[100];
                CSFS_lambda=para.CSFS_lambda;
                [ W_CSFS] = CSFS_learn(X_tr,Y_tr,CSFS_lambda);
                normW_CSFS = sqrt(sum(W_CSFS.*W_CSFS,2));
                [W_CSFS_Weight, W_CSFS_sorted_features] = sort(-normW_CSFS);
                percent = 0.9;
                Num_SelectFea = floor(percent*d);       
                W_CSFS_SelectFeaIdx{i} = W_CSFS_sorted_features(1:Num_SelectFea,:);
                CSFSXTrain = X_tr(:,W_CSFS_SelectFeaIdx{i});
                CSFSXTest = X_te(:,W_CSFS_SelectFeaIdx{i});   
                CompareXTrain = CSFSXTrain;
                CompareXTest = CSFSXTest;
                saveParameter=num2str(para.CSFS_lambda);
            case 'LSG21'
              %% LSG21  2014
                % LSG21 parameter
                para.LSG21_lambda=1000;
                [W_LSG21 ] = L2G21_new(X_tr,Y_tr, para.LSG21_lambda); %%%%%%%%%%%%%   LSG21_lambda 

                normW_LSG21 = sqrt(sum(W_LSG21.*W_LSG21,2));
                [W_LSG21_Weight, W_LSG21_sorted_features] = sort(-normW_LSG21);
                percent = 0.9;
                Num_SelectFea = floor(percent*d);       
                W_LSG21_SelectFeaIdx{i} = W_LSG21_sorted_features(1:Num_SelectFea,:);
                LSG21XTrain = X_tr(:,W_LSG21_SelectFeaIdx{i});
                LSG21XTest = X_te(:,W_LSG21_SelectFeaIdx{i}); 
                CompareXTrain = LSG21XTrain;
                CompareXTest = LSG21XTest;
                saveParameter=num2str(para.LSG21_lambda);
            case 'SLRR'
              %%  2013 SLRR      LeastSquareLDAR21
                % SLRR parameter
                para.SLRR_lambda=0.001;
                
                para.r = min(n,label)*1;
                
                [B,A] = LeastSquareLDAR21(X_tr,Y_tr, para.SLRR_lambda, para.r);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    SLRR_lambda     r
                W=B*A;
                normB = sqrt(sum(W.*W,2));
                [LDAR21_Weight, LDAR21_sorted_features] = sort(-normB);
                percent = 0.9;
                Num_SelectFea = floor(percent*d);       
                LDAR21_SelectFeaIdx{i} = LDAR21_sorted_features(1:Num_SelectFea,:);
                LDAR21XTrain = X_tr(:,LDAR21_SelectFeaIdx{i});
                LDAR21XTest = X_te(:,LDAR21_SelectFeaIdx{i});
                CompareXTrain = LDAR21XTrain;
                CompareXTest = LDAR21XTest;
                saveParameter=[num2str(para.SLRR_lambda),'_',num2str(para.r)];
            case 'RSR'
              %%  2015 RSR      
                % RSR parameter
                para.r = min(n,label)*1;
                
                [W,~] = L21R21(X_tr,Y_tr,para.r);
                normB = sqrt(sum(W.*W,2));
                [LDAR21_Weight, LDAR21_sorted_features] = sort(-normB);
                percent = 0.9;
                Num_SelectFea = floor(percent*d);       
                LDAR21_SelectFeaIdx{i} = LDAR21_sorted_features(1:Num_SelectFea,:);
                LDAR21XTrain = X_tr(:,LDAR21_SelectFeaIdx{i});
                LDAR21XTest = X_te(:,LDAR21_SelectFeaIdx{i});
                CompareXTrain = LDAR21XTrain;
                CompareXTest = LDAR21XTest;
                saveParameter=num2str(para.r);
            case 'MSFS'
              %% MSFS   2014
              para.MSFSlambda1 =10;
              para.MSFSlambda2 =10;
              para.MSFSlambda3 =10;
              
               [W_MSFS]= LF3L21(X_tr,Y_tr,para.MSFSlambda1,para.MSFSlambda2,para.MSFSlambda3);
                normW_MSFS = sqrt(sum(W_MSFS.*W_MSFS,2));
                [W_MSFS_Weight, W_MSFS_sorted_features] = sort(-normW_MSFS);
                percent = 0.9;
                Num_SelectFea = floor(percent*d);       
                W_MSFS_SelectFeaIdx{i} = W_MSFS_sorted_features(1:Num_SelectFea,:);
                MSFSXTrain = X_tr(:,W_MSFS_SelectFeaIdx{i});
                MSFSXTest = X_te(:,W_MSFS_SelectFeaIdx{i});
                CompareXTrain = MSFSXTrain;
                CompareXTest = MSFSXTest;
                saveParameter=[num2str(para.MSFSlambda1),'_',num2str(para.MSFSlambda2),'_',num2str(para.MSFSlambda3)];
                
        end



    
%% SVR:Support Vector Regression
    [Result.aCC(j,i),Result.aRMSE(j,i),maxind(j,i)] = DD_SVR1(CompareXTrain,CompareXTest,Y_tr,Y_te);
    
    end 
end 
meanaCC = mean(Result.aCC);
meanaRMSE = mean(Result.aRMSE);
save(['compare_result\',dataname,'_',compareName,'_',saveParameter,'_',num2str(meanaCC),'.mat'],'Result','Ttest','para','meanaCC','meanaRMSE');     

   
       