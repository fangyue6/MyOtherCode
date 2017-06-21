% function Demo_f2L21(dataset,libsvmType)
%%%%%%%%%%%% NOTE THAT %%%%%%%%%%%%%%%%%%%
close all;
clc;clear;
%  before runing the demo, you should have installed LIBSVM and change  the following  path:
 addpath(genpath('F:\Program Files\MATLAB\MATLAB Production Server\R2015a\toolbox\libsvm-3.20\matlab'));
 addpath(genpath('D:\fangyue\algorithm\feature-select-2'));
% 
% 
warning('off')

%% 定义参数start
folderPath='D:\fangyue\algorithm\feature-select-2\';
datasetsPath=[folderPath,'datasets\'];
% resultPath=[folderPath,'result\compare\'];
resultPath=[folderPath,'result\our\'];
%'chess_uni','train','Ecoli8','ALLAML','DBWorld','GLI-85','lung','pixraw10P','Yale15','ORL40','umist','COIL20','PCMAC'
document = {'PCMAC'};
%要跑的对比算法,跑什么算法就写上对应的算法名字就好
% 此算法的名字  xijAB_ABS
%'CSFS','LDA','LR', 'RSR' ,'jelsr','FSRobust_ALM','RPCA_OM','traceratioFS_unsupervised','FLGPP'
%xijAB_ABS   LDA   LR   RSR   FSASL
%CSFS RSR LR 比较快
algorithm ={'LHSL_FS'};
parm_r=1;
%%定义参数end

%% 循环数据集 start
for d = 1:length(document)
    
    %% 循环对比算法 start
    for algorithmIndex = 1:length(algorithm)
        
        fileName = [datasetsPath,char(document(d)) '.mat'];
        file = load(fileName);

        %与数据对应的类数
        classnum = length(unique(file.Y));
        if classnum==2
            file.Y(file.Y==-1)=2;
        end
        if size(file.X,2)>400
            file.X=file.X(:,1:400);
        end
%         if size(file.X,1)>1000
%             file.X=file.X(1:1000,:);
%             file.Y=file.Y(1:1000,:);
%         end
        
        [m n]=size(file.X);
        X=full(file.X);
        %X=selectX(X,0.5);
        %  normalize each row to unit
        %X = X./repmat(sqrt(sum(X.^0,2)),1,size(X,2));
        %  normalize each column to unit
        %X = X./repmat(sqrt(sum(X.^2,1)),size(X,1),1);
        X = NormalizeFea(X,0);
        clear ans info
        label=file.Y;

        pars = [];
        pars.classnum=classnum;%类别数目

      
          pars.lambda1=[10^0];
          pars.lambda2=[10^1];
%         pars.lambda1 = [10^-3,10^-2,10^-1,10^0];
%         pars.lambda2 = [10^1,10^2,10^3];
        
        pars.r = min(m,n)*parm_r;
        pars.cvk = 5;  %交叉验证的次数   svm训练参数   
        pars.k = 2;
        pars.Ite = 20;
        pars.c = [1];
        pars.g = [2];
%         pars.c = -10:2:10;
%         pars.g = -10:2:10;
        pars.epsilon = 0.0001;
        libsvmType='liblinear';
        pars.libsvmType = libsvmType;
        
        pars.algorithm = char(algorithm(algorithmIndex));
    

        for k = 1:pars.k
            disp([num2str(k),'折']);
            for labelIndex=1:classnum
                ind(:,k) = crossvalind('Kfold',size(find(label),1),pars.k);
                %ind(:,k) = crossvalind('Kfold',size(find(label == labelIndex),1),pars.k);
            end
            
            %cross-validation
            test = (ind(:,k) == k); train = ~test;
            
            
            
           alphas=[0.1];
           betas=[1];
           ps =[2];
           
           for ialpha=1:length(alphas)
               pars.alpha = alphas(ialpha);
               for ibeta=1:length(betas)
                   pars.beta = betas(ibeta);
                   for ip=1:length(ps)
                       pars.p = ps(ip);
                       [result_opt(k,:),testResults(k,:),mseResults(k,:)] = Step2_our_21_2p_FS(X(train,:),X(test,:),label(train,:),label(test,:),pars,k,char(document(d))); 
                   end
               end
           end
            
        %     iii = iii + k;
        %     waitbar(iii/steps);
        end

        % close(h)
        meantestresult = mean(testResults)
        meanmseresutlt = mean(mseResults)
%         if(strcmp(pars.algorithm,'LHSL_FS'))
%         	save([resultPath,char(document(d)),'_',pars.algorithm,'_',num2str(meantestresult),'% -',num2str(pars.lambda1(1)),'-',num2str(pars.lambda2(1)),'.mat']);
%         else 
%             save([resultPath,char(document(d)),'_',pars.algorithm,'_',num2str(meantestresult),'%.mat']);
%         end
        %data ={'email','854289665@qq.com','subject',[char(document(d)),'_',pars.algorithm],'content',[char(document(d)),'_',pars.algorithm,'_',num2str(meantestresult)]};
        %urlread('http://172.16.25.68:8080/Mail/mail','POST',data);
        

    end
    %%循环对比算法 end
    
end
%%循环数据集 end

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






