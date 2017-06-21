clc;
clear all;
% addpath(genpath('E:\MATLAB\R2014a\toolbox\libsvm-3.20\matlab'));


dataname = 'WQ';
load(['D:\fangyue\algorithm\regression\dataset\',dataname]);%加载数据集


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
       alpha = 3;
       beta = -2;
       p = 1.2;
       para.alpha = 10^alpha;%调节参数（10^-5 -- 10^5）
       para.beta = 10^beta;%调节参数（10^-5 -- 10^5）
       para.r = min(n,label)*1;
       para.p = p;%调节参数  0-2

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



%% SVR:Support Vector Regression

  [LY.aCC(j,i),LY.aRMSE(j,i),LY.maxind(j,i)] =DD_SVR1(Y_tr,Y_te,LYXTrain,LYXTest);

    end 
end 
meanaCC = mean(LY.aCC);
meanaRMSE = mean(LY.aRMSE);
save([dataname,'_',num2str(alpha),'_',num2str(beta),'_',num2str(p),'_',num2str(meanaCC),'.mat'],'LY','Ttest','para','meanaCC','meanaRMSE');     

   
       