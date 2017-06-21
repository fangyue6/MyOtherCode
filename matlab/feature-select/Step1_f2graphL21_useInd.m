% function Demo_f2L21(dataset,libsvmType)
%%%%%%%%%%%% NOTE THAT %%%%%%%%%%%%%%%%%%%
close all;
clc;clear;
%  before runing the demo, you should have installed LIBSVM and change  the following  path:
 addpath(genpath('F:\Program Files\MATLAB\MATLAB Production Server\R2015a\toolbox\libsvm-3.20\matlab'));
% 
% 

%% 定义参数start
folderPath='D:\fangyue\algorithm\feature-select\';
datasetsPath=[folderPath,'result\compare\利用\'];
resultPath=[folderPath,'result\compare1\'];
document={'ALLAML_CSFS_90.8929%'}
%'DBWorld'  'ALLAML' 'PCMAC' 'GLI-85' 'train' 'Yale15' 'ORL40' 'pixraw10P' 'lung'


% 'RSR' 'jelsr' 'CSFS' 'FSASL' 'FSSI' 'RPCA_OM' 'xijAB_ABS'

        d=1;
        fileName = [datasetsPath,char(document(d)) '.mat'];
        load(fileName);


    
        % % steps = 13; % should be changed by your code
        for k = 1:pars.k
%             disp([num2str(k),'折']);
%             for labelIndex=1:classnum
%                 ind(:,k) = crossvalind('Kfold',size(find(label),1),pars.k);
%                 %ind(:,k) = crossvalind('Kfold',size(find(label == labelIndex),1),pars.k);
%             end
            %cross-validation
            test = (ind(:,k) == k); train = ~test;
            [opt(k,:),pred_label(k,:),testResults(k,:),mseResults(k,:)] = Step2_D_train_cv(X(train,:),X(test,:),label(train,:),label(test,:),pars,k,char(document(d))); 
        end

        % close(h)
        meantestresult = mean(testResults)
        meanmseresutlt = mean(mseResults)
        if(strcmp(pars.algorithm,'xijAB_ABS'))
        	save([resultPath,char(document(d)),'_',pars.algorithm,'_',num2str(meantestresult),'% -',num2str(pars.lambda1(1)),'-',num2str(pars.lambda2(1)),'.mat']);
        else 
            save([resultPath,char(document(d)),'_',pars.algorithm,'_',num2str(meantestresult),'%.mat']);
        end






