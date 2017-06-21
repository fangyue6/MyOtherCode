% function Demo_f2L21(dataset,libsvmType)
%%%%%%%%%%%% NOTE THAT %%%%%%%%%%%%%%%%%%%
close all;
clc;clear all;
%  before runing the demo, you should have installed LIBSVM and change  the following  path:
 addpath(genpath('F:\Program Files\MATLAB\MATLAB Production Server\R2015a\toolbox\libsvm-3.20\matlab'));
 addpath(genpath('D:\fangyue\algorithm\feature-select-2'));
% 
% 
warning('off')

%% 定义参数start
folderPath1='D:\fangyue\algorithm\feature-select-2\result\our\low_rank\';
datasetsPath1=folderPath1;
% resultPath=[folderPath,'result\compare\'];
resultPath1=[folderPath1,'result\'];

%'chess_uni','train','Ecoli8','ALLAML','DBWorld','GLI-85','lung','pixraw10P','Yale15','ORL40','umist','COIL20'
document1 = {'COIL20_LHSL_FS_97.7083% -1-10'};

algorithm ={'LHSL_FS'};
%%定义参数end
rs = [1 3 5 7 9 11 13 15 17]
for ir=1:length(rs)

%% 循环数据集 start
for d = 1:length(document1)
    
    %% 循环对比算法 start
    for algorithmIndex = 1:length(algorithm)
        
        fileName = [datasetsPath1,char(document1(d)) '.mat'];
        load(fileName);


        
        pars.r = rs(ir);
        
        pars.cvk = 5;  %交叉验证的次数   svm训练参数   
        pars.k = 10;
        pars.Ite = 20;

        pars.epsilon = 0.0001;
        libsvmType='liblinear';
        pars.libsvmType = libsvmType;
    

    

        for k = 1:pars.k
            disp([num2str(k),'折 - ',num2str(ir)]);
%             for labelIndex=1:classnum
%                 ind(:,k) = crossvalind('Kfold',size(find(label),1),pars.k);
%                 %ind(:,k) = crossvalind('Kfold',size(find(label == labelIndex),1),pars.k);
%             end
            
            pars.c  = result_opt(k).bestc;
            pars.g  = result_opt(k).bestg;

            %cross-validation
            test = (ind(:,k) == k); train = ~test;
            
            
            
           pars.alpha = result_opt(k).alpha;
           pars.beta = result_opt(k).beta;
           pars.p = result_opt(k).p;
           [low_rank_result_opt(k,:),testResults1(k,:),mseResults1(k,:)] = Step2_our_21_2p_FS(X(train,:),X(test,:),label(train,:),label(test,:),pars,k,char(document(d))); 
           
           
        end

        meantestresult = mean(testResults1)
        meanmseresutlt = mean(mseResults1)
        save([resultPath1,char(document(d)),'-',pars.algorithm,'_',num2str(meantestresult),'%-r',num2str(pars.r),'.mat']);
        %data ={'email','854289665@qq.com','subject',[char(document(d)),'_',pars.algorithm],'content',[char(document(d)),'_',pars.algorithm,'_',num2str(meantestresult)]};
        %urlread('http://172.16.25.68:8080/Mail/mail','POST',data);
        
    end
    %%循环对比算法 end
    

end
%%循环数据集 end

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






