% function Demo_f2L21(dataset,libsvmType)
%%%%%%%%%%%% NOTE THAT %%%%%%%%%%%%%%%%%%%
close all;
clc;clear all;
%  before runing the demo, you should have installed LIBSVM and change  the following  path:
 addpath(genpath('F:\Program Files\MATLAB\MATLAB Production Server\R2015a\toolbox\libsvm-3.20\matlab'));
 addpath(genpath('D:\fangyue\algorithm\feature-select'));
% 
% 
warning('off')

%% 定义参数start
folderPath1='D:\fangyue\algorithm\feature-select\result\parameter_sensitivity\';
datasetsPath1=folderPath1;
% resultPath=[folderPath,'result\compare\'];
resultPath1=[folderPath1,'result\'];

%'chess_uni','train','Ecoli8','ALLAML','DBWorld','GLI-85','lung','pixraw10P','Yale15','ORL40','umist','COIL20'
document1 = {'train_xijAB_ABS_84.9265%','warpAR10P_xijAB_ABS_91.5385% -0.01-10','Yale15_xijAB_ABS_81.2868%'};
% 'ALLAML_xijAB_ABS_95.7143%','chess_uni_xijAB_ABS_99.75% -1-10','DBWorld-30%-R90%-700_xijAB_ABS_84.5238%',
%'GLI-85_xijAB_ABS_94.1667%','lung_xijAB_ABS_95.5952%','ORL40_xijAB_ABS_97% -0.001-10',
%'Parkinsons2_xijAB_ABS_87.6842% -1-10','PCMAC_xijAB_ABS_92.6932% -1-10','pixraw10P_xijAB_ABS_97%',
%'train_xijAB_ABS_84.9265%','warpAR10P_xijAB_ABS_91.5385% -0.01-10','Yale15_xijAB_ABS_81.2868%'

algorithm ={'xijAB_ABS'};
%%定义参数end

%% 循环数据集 start
for  dd= 1:length(document1)
        
        fileName = [datasetsPath1,'data/',char(document1(dd)) '.mat'];
        load(fileName);
        
        [n,m] = size(X);
        if(m>400)
            X=X(:,1:400);
        end
        


        
        pars.r = min(m,n);
        
        pars.cvk = 5;  %交叉验证的次数   svm训练参数   
        pars.k = 10;
        pars.Ite = 20;

        pars.epsilon = 0.0001;
        libsvmType='liblinear';
        pars.libsvmType = libsvmType;
 
           alphas=[-2 :1:3];
           betas=[-2:1:3];
           meantestresult=[];
           meanmseresutlt=[];
           for ialpha=1:length(alphas)
               pars.lambda1 = 10^alphas(ialpha);
               for ibeta=1:length(betas)
                   pars.lambda2 = 10^betas(ibeta);
                   
                   for labelIndex=1:classnum
                        ind = crossvalind('Kfold',size(find(label),1),pars.k);
                   end

                   pars.c  = [1];
                   pars.g  = [3];

                   %cross-validation
                   test = (ind == k); train = ~test;

                   [qqqq,wwwww,testResults,mseResults] = Step2_D_train_cv(X(train,:),X(test,:),label(train,:),label(test,:),pars,k,char(document(dd))); 

                  meantestresult(ialpha,ibeta) = testResults
                  meanmseresutlt(ialpha,ibeta) = mseResults
                   
               end
           end
           save([resultPath1,char(document(dd)),'.mat']);

end

% FSSI
% Yi Yang, Zhigang Ma, Alex Hauptmann and Nicu Sebe   2013
% Feature Selection for Multimedia Analysis by Sharing Information among Multiple Tasks
% 
% 
% FSASL
% Liang Du,Yi-Dong Shen  2015
% Unsupervised Feature Selection with Adaptive Structure Learning
% 
% 
% UDFS
% Yi Yang, Heng Tao Shen, Zhigang Ma, Zi Huang, Xiaofang Zhou   2011
% l2,1-Norm Regularized Discriminative Feature Selection for Unsupervised Learning
% 
% 
% MCFS
% Deng Cai ,Chiyuan Zhang ,Xiaofei He   2010
% Unsupervised Feature Selection for Multi-Cluster Data
% 
% 
% NSDR
% Yi Yang, Heng Tao Shen, Feiping Nie, Rongrong Ji, Xiaofang Zhou   2011
% Nonnegative Spectral Clustering with Discriminative Regularization
% 
% 
% CSFS
% Xiaojun Chang, Feiping Nie, Yi Yang and Heng Huang       AAAI 2014.
% A Convex Formulation for Semi-Supervised Multi-Label Feature Selection. 
% 
% 
% FRFS0  
% W Wang，H Zhang，P Zhu，D Zhang，W Zuo      2015
% Non-convex Regularized Self-representation for Unsupervised Feature Selection
% 
% 
% RFS
% Feiping Nie, Heng Huang, Xiao Cai, Chris Ding.    2010
% Efficient and Robust Feature Selection via Joint L21-Norms Minimization
% 
% 
% RSR
% Feiping Nie, Heng Huang, Xiao Cai, Chris Ding.   2015
% Efficient and Robust Feature Selection via Joint L21-Norms Minimization  
% 
% 
% LSG21
% Xiao Cai, Feiping Nie, Weidong Cai, Heng Huang   2013/2014
% New Graph Structured Sparsity Model for Multi-Label Image Annotations
% 
% 
% SFUS
% Zhigang Ma, Feiping Nie, Yi Yang, Jasper Uijlings, and Nicu Sebe  2012
% Web Image Annotation via Subspace-Sparsity Collaborated Feature Selection






