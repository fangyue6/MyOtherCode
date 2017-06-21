clc;clear;
% addpath(genpath('F:\matlab\toolbox\libsvm-3.20\matlab'));
% addpath(genpath('F:\matlab\toolbox\libsvm-3.20'));
bestresult=[0,0.5];
dataname='oes97';
% CSFS LSG21 SLRR RSR MSFS
Compare_group={ 'RSR'};%'CSFS' 'LSG21'   'RSR' 'MSFS'
for CoGp=1:1:length(Compare_group)
    compareName=Compare_group{CoGp};
    load(['D:\fangyue\algorithm\regression\dataset\',dataname,'.mat']);
%     fea = data;
%     gnd = label;
    fea = NormalizeFea(fea,0);
    gnd = NormalizeFea(gnd,0);
    [n, d] = size(fea);
%     if d>50
%         d=50;
%     end
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
    switch compareName
        case 'CSFS'
            %% CSFS     2014-semi_supervised
            % CSFS parameter
            for lambda_CSFS=power(10,-3:1:3)
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
                        para.CSFS_lambda=lambda_CSFS;
                        CSFS_lambda=para.CSFS_lambda;
                        [ W_CSFS] = CSFS_learn(X_tr,Y_tr,CSFS_lambda);
                        normW_CSFS = sqrt(sum(W_CSFS.*W_CSFS,2));
                        [W_CSFS_Weight, W_CSFS_sorted_features] = sort(-normW_CSFS);
                        percent = 1;
                        Num_SelectFea = floor(percent*d);
                        W_CSFS_SelectFeaIdx{i} = W_CSFS_sorted_features(1:Num_SelectFea,:);
                        CSFSXTrain = X_tr(:,W_CSFS_SelectFeaIdx{i});
                        CSFSXTest = X_te(:,W_CSFS_SelectFeaIdx{i});
                        CompareXTrain = CSFSXTrain;
                        CompareXTest = CSFSXTest;
                        %% SVR:Support Vector Regression
                        [Result.aCC(j,i),Result.aRMSE(j,i),maxind(j,i)] = DD_SVR1(CompareXTrain,CompareXTest,Y_tr,Y_te);
                        
                    end
                end
                meanaCC = mean(Result.aCC);
                meanaRMSE = mean(Result.aRMSE);

                saveParameter=num2str(para.CSFS_lambda);
                save(['compare_result\',dataname,'_',compareName,'_',saveParameter,'_',num2str(meanaCC),'.mat'],'Result','Ttest','para','meanaCC','meanaRMSE');
                save('tmp','n','d','label','fea','gnd','lambda','dataname','Compare_group','compareName','bestresult');
                clear all;
                load('tmp');
                disp('------------------  CSFS  ------------------');
            end
        case 'LSG21'
            %% LSG21  2014
            % LSG21 parameter
            for lambda_LSG21=power(10,-3:1:3)
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
                        para.LSG21_lambda=lambda_LSG21;
                        [W_LSG21 ] = L2G21_new(X_tr,Y_tr, para.LSG21_lambda); %%%%%%%%%%%%%   LSG21_lambda
                        
                        normW_LSG21 = sqrt(sum(W_LSG21.*W_LSG21,2));
                        [W_LSG21_Weight, W_LSG21_sorted_features] = sort(-normW_LSG21);
                        percent = 1;
                        Num_SelectFea = floor(percent*d);
                        W_LSG21_SelectFeaIdx{i} = W_LSG21_sorted_features(1:Num_SelectFea,:);
                        LSG21XTrain = X_tr(:,W_LSG21_SelectFeaIdx{i});
                        LSG21XTest = X_te(:,W_LSG21_SelectFeaIdx{i});
                        CompareXTrain = LSG21XTrain;
                        CompareXTest = LSG21XTest;
                        %% SVR:Support Vector Regression
                        [Result.aCC(j,i),Result.aRMSE(j,i),maxind(j,i)] = DD_SVR1(CompareXTrain,CompareXTest,Y_tr,Y_te);
                        
                    end
                end
                meanaCC = mean(Result.aCC);
                meanaRMSE = mean(Result.aRMSE);
                saveParameter=num2str(para.LSG21_lambda);
                save(['compare_result\',dataname,'_',compareName,'_',saveParameter,'_',num2str(meanaCC),'.mat'],'Result','Ttest','para','meanaCC','meanaRMSE');
                save('tmp','n','d','label','fea','gnd','lambda','dataname','Compare_group','compareName','bestresult');
                clear all;
                load('tmp');
                disp('------------------  LSG21  ------------------');
            end
        case 'SLRR'
            %%  2013 SLRR      LeastSquareLDAR21
            % SLRR parameter
            for lambda_SLRP=power(10,-3:1:3)
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
                        para.SLRR_lambda=lambda_SLRP;
                        
                        para.r = min(n,label)*1;
                        
                        [B,A] = LeastSquareLDAR21(X_tr,Y_tr, para.SLRR_lambda, para.r);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    SLRR_lambda     r
                        W=B*A;
                        normB = sqrt(sum(W.*W,2));
                        [LDAR21_Weight, LDAR21_sorted_features] = sort(-normB);
                        percent = 1;
                        Num_SelectFea = floor(percent*d);
                        LDAR21_SelectFeaIdx{i} = LDAR21_sorted_features(1:Num_SelectFea,:);
                        LDAR21XTrain = X_tr(:,LDAR21_SelectFeaIdx{i});
                        LDAR21XTest = X_te(:,LDAR21_SelectFeaIdx{i});
                        CompareXTrain = LDAR21XTrain;
                        CompareXTest = LDAR21XTest;
                        
                        %% SVR:Support Vector Regression
                        [Result.aCC(j,i),Result.aRMSE(j,i),maxind(j,i)] = DD_SVR1(CompareXTrain,CompareXTest,Y_tr,Y_te);
                        
                    end
                end
                meanaCC = mean(Result.aCC);
                meanaRMSE = mean(Result.aRMSE);
                saveParameter=[num2str(para.SLRR_lambda),'_',num2str(para.r)];
                save(['compare_result\',dataname,'_',compareName,'_',saveParameter,'_',num2str(meanaCC),'.mat'],'Result','Ttest','para','meanaCC','meanaRMSE');
                Para_r= para.r;
                save('tmp','n','d','label','fea','gnd','lambda','dataname','Compare_group','compareName','bestresult','Para_r');
                clear all;
                load('tmp');
                para.r=Para_r;
                disp('------------------  SLRR  ------------------');
            end
        case 'RSR'
            %%  2015 RSR
            % RSR parameter
            for k=1
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
%                         para.r = floor(min(n,label)*k);
                        para.r=4;
                        
                        [W,~] = L21R21(X_tr,Y_tr,para.r);
                        normB = sqrt(sum(W.*W,2));
                        [LDAR21_Weight, LDAR21_sorted_features] = sort(-normB);
                        percent =0.1;
                        Num_SelectFea = floor(percent*d);
%                         Num_SelectFea=10;
                        LDAR21_SelectFeaIdx{i} = LDAR21_sorted_features(1:Num_SelectFea,:);
                        LDAR21XTrain = X_tr(:,LDAR21_SelectFeaIdx{i});
                        LDAR21XTest = X_te(:,LDAR21_SelectFeaIdx{i});
                        CompareXTrain = LDAR21XTrain;
                        CompareXTest = LDAR21XTest;
                        
                        %% SVR:Support Vector Regression
                        [Result.aCC(j,i),Result.aRMSE(j,i),maxind(j,i)] = DD_SVR1(CompareXTrain,CompareXTest,Y_tr,Y_te);
                        
                    end
                end
                meanaCC = mean(Result.aCC);
                meanaRMSE = mean(Result.aRMSE);
                saveParameter=num2str(para.r);
                save(['compare_result\',dataname,'_',compareName,'_',saveParameter,'_',num2str(meanaCC),'.mat'],'Result','Ttest','para','meanaCC','meanaRMSE');
                save('tmp','n','d','label','fea','gnd','lambda','dataname','Compare_group','compareName','bestresult');
                clear all;
                load('tmp');
                disp('------------------  RSR  ------------------');
            end
        case 'MSFS'
            %% MSFS   2014
            for lambda_MSFS1=power(10,[1 2 3])
                for lambda_MSFS2=power(10,[1 2 3])
                    for lambda_MSFS3=power(10,[ 1 2 3])
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
                                para.MSFSlambda1 =lambda_MSFS1;
                                para.MSFSlambda2 =lambda_MSFS2;
                                para.MSFSlambda3 =lambda_MSFS3;
                                
                                [W_MSFS]= LF3L21(X_tr,Y_tr,para.MSFSlambda1,para.MSFSlambda2,para.MSFSlambda3);
                                normW_MSFS = sqrt(sum(W_MSFS.*W_MSFS,2));
                                [W_MSFS_Weight, W_MSFS_sorted_features] = sort(-normW_MSFS);
                                percent = 1;
                                Num_SelectFea = floor(percent*d);
                                W_MSFS_SelectFeaIdx{i} = W_MSFS_sorted_features(1:Num_SelectFea,:);
                                MSFSXTrain = X_tr(:,W_MSFS_SelectFeaIdx{i});
                                MSFSXTest = X_te(:,W_MSFS_SelectFeaIdx{i});
                                CompareXTrain = MSFSXTrain;
                                CompareXTest = MSFSXTest;
                                
                                %% SVR:Support Vector Regression
                                [Result.aCC(j,i),Result.aRMSE(j,i),maxind(j,i)] = DD_SVR1(CompareXTrain,CompareXTest,Y_tr,Y_te);
                                
                            end
                        end
                        meanaCC = mean(Result.aCC);
                        meanaRMSE = mean(Result.aRMSE);
                        saveParameter=[num2str(para.MSFSlambda1),'_',num2str(para.MSFSlambda2),'_',num2str(para.MSFSlambda3)];
                        save(['compare_result\',dataname,'_',compareName,'_',saveParameter,'_',num2str(meanaCC),'.mat'],'Result','Ttest','para','meanaCC','meanaRMSE');
                        save('tmp','n','d','label','fea','gnd','lambda','dataname','Compare_group','compareName','bestresult');
                        clear all;
                        load('tmp');
                        disp('------------------  MSFS  ------------------');
                    end
                end
            end
    end
end
