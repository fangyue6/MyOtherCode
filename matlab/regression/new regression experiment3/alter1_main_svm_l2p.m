clc;
clear;
% addpath(genpath('E:\MATLAB\R2014a\toolbox\libsvm-3.20\matlab'));



load('D:\fangyue\algorithm\regression\dataset\oes97');%加载数据集

% 
% fea = data;%%%%%%%%%%%%%%%%%%%%%
% gnd = label;%%%%%%%%%%%%%%%%%%
fea = NormalizeFea(fea,0);
gnd = NormalizeFea(gnd,0);
% fea = stereo;
% gnd = stereolabel;
% 
[n, d] = size(fea);
label = size(gnd,2);
bestresult=[0,0];

%% LY objective function
for k=[1]
      para.r = min(n,label)*k;
      para.r=10;
%       para.r = 3;
%     Tstr1=['K= ',num2str(k)];
%     disp(Tstr1);
    for p=[1]
        para.p=p;
%         Tstr2=['P= ',num2str(p)];
%         disp(Tstr2);
        for alpha=power(10,[-1])
            para.alpha=alpha;
%                 Tstr4=['alpha= ',num2str(alpha)];
%                 disp(Tstr4);
            for beta=power(10,[1])
                para.beta=beta;
    %             Tstr3=['beta= ',num2str(beta)];
    %             disp(Tstr3);


%% start regression
lambda = 10;
for j=1:length(lambda)
             ind = crossvalind('Kfold',n,10);

    for i = 1:10
        test = (ind==i);
        train = ~test;
        X_tr = fea(train,:);
        X_te = fea(test,:);
        Y_tr = gnd(train,:);
        Y_te = gnd(test,:);
        Ttest(j,i) = {ind==i};


 %% fangyue objective function
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
                result=[mean(LY.aCC),mean(LY.aRMSE)];
                if result(1,1)>bestresult(1,1)||result(1,2)<bestresult(1,2)
                   bestresult=result;
                   a_Alpha=log10(para.alpha);
                   b_Beta=log10(para.beta);
                   bestparameter=['_',num2str(a_Alpha),'_',num2str(b_Beta),'_',num2str(para.r),'_',num2str(p),'_',num2str(meanaCC)];
                   save(['oes97_',bestparameter,'.mat'],'LY','Ttest','para','meanaCC','meanaRMSE');
                end
                save('tmp','n','d','label','alpha','beta','k','p','fea','gnd','bestresult');
                clear all;
                load('tmp');
                para.beta=beta;
                para.alpha=alpha;
                para.r = min(n,label)*k;
                para.p=p;
            end
        end
    end
end
   
       