% function Demo_f2L21(dataset,libsvmType)
%%%%%%%%%%%% NOTE THAT %%%%%%%%%%%%%%%%%%%
close all;
clc;clear;
%  before runing the demo, you should have installed LIBSVM and change  the following  path:
 addpath(genpath('D:\Program Files\MATLAB\MATLAB Production Server\R2015a\toolbox\libsvm-3.20\matlab'));
% 
% 


load('datasets\solar_uni.mat');
[m n]=size(X);
% q = randperm(n,4000);
% X=X(:,q);
X=full(X);
clear ans info
%  X=X(:,1:5000);
% Y1=load('D:\data\dataset\le song\icml_data\data20\y.txt');
%  Y1=Y1+1;
label=Y;
i=0;
j=1;
pars.lambda1 = [10^i];
pars.lambda2 = [10^j];
% pars.lambda3 = [10^2];
% pars.u = [1000 10000 1000000];
pars.r = min(m,n);
pars.cvk = 5;
pars.k = 10;
pars.Ite = 20;
pars.c = -10:2:10;
pars.g = -10:2:10;
pars.epsilon = 0.0001;
libsvmType='liblinear'
pars.libsvmType = libsvmType;

% % h = waitbar(0,'Please wait......');
% % iii = 0;
% % steps = 10; % should be changed by your code
for k = 1:pars.k
    ind1(:,k) = crossvalind('Kfold',size(find(label == 1),1),pars.k);
    ind2(:,k) = crossvalind('Kfold',size(find(label == 2),1),pars.k);
    ind3(:,k) = crossvalind('Kfold',size(find(label == 3),1),pars.k);
    ind4(:,k) = crossvalind('Kfold',size(find(label == 4),1),pars.k);
    ind5(:,k) = crossvalind('Kfold',size(find(label == 5),1),pars.k);
    ind6(:,k) = crossvalind('Kfold',size(find(label == 6),1),pars.k);
%     ind7(:,k) = crossvalind('Kfold',size(find(label == 7),1),pars.k);
%     ind8(:,k) = crossvalind('Kfold',size(find(label == 8),1),pars.k);
%     ind9(:,k) = crossvalind('Kfold',size(find(label == 9),1),pars.k);
%     ind10(:,k) = crossvalind('Kfold',size(find(label == 10),1),pars.k);
%      ind11(:,k) = crossvalind('Kfold',size(find(label == 11),1),pars.k);
%     ind12(:,k) = crossvalind('Kfold',size(find(label == 12),1),pars.k);
%     ind13(:,k) = crossvalind('Kfold',size(find(label == 13),1),pars.k);
%     ind14(:,k) = crossvalind('Kfold',size(find(label == 14),1),pars.k);
%     ind15(:,k) = crossvalind('Kfold',size(find(label == 15),1),pars.k);
%     ind = [ind1;ind2;ind3;ind4;ind5;ind6;ind7;ind8;ind9;ind10;ind11;ind12;ind13;ind14;ind15];
     ind = [ind1;ind2;ind3;ind4;ind5;ind6];
    %cross-validation
    test = (ind(:,k) == k); train = ~test;
    [testResults(k,:),mseResults(k,:)] = D_train_cv(X(train,:),X(test,:),...
        label(train,:),label(test,:),pars); 
%     iii = iii + k;
%     waitbar(iii/steps);
end
% close(h)
meantestresult = mean(testResults)
meanmseresutlt = mean(mseResults)
path=['solar_uni_lam1_10_',num2str(i),'_lam2_10_',num2str(j)];
save(path);