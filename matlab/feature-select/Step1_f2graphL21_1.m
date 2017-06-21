% function Demo_f2L21(dataset,libsvmType)
%%%%%%%%%%%% NOTE THAT %%%%%%%%%%%%%%%%%%%
close all;
clc;clear;
%  before runing the demo, you should have installed LIBSVM and change  the following  path:
 addpath(genpath('F:\Program Files\MATLAB\MATLAB Production Server\R2015a\toolbox\libsvm-3.20\matlab'));
% 
% 

%% �������start
folderPath='D:\fangyue\algorithm\feature-select\';
datasetsPath=[folderPath,'datasets\'];
resultPath=[folderPath,'result\compare\'];
document = {'gene_17'};%,'solar_uni' 'umist','chess_uni'     'Forest4','Parkinsons2','SPECTF_Heart2','HillValley_uni','pixraw10P','PCMAC'
%Ҫ�ܵĶԱ��㷨,��ʲô�㷨��д�϶�Ӧ���㷨���־ͺ�
% ���㷨������  xijAB_ABS
%'CSFS','LDA','LR', 'RSR' ,'jelsr','FSRobust_ALM','RPCA_OM','traceratioFS_unsupervised','FLGPP'
%xijAB_ABS   LDA   LR   RSR
%CSFS RSR LR �ȽϿ�
algorithm ={'LS21'};
parm_r=1;
%%�������end

%% ѭ�����ݼ� start
for d = 1:length(document)
    
    %% ѭ���Ա��㷨 start
    for algorithmIndex = 1:length(algorithm)
        
        fileName = [datasetsPath,char(document(d)) '.mat'];
        file = load(fileName);

        %�����ݶ�Ӧ������
        classnum = length(unique(file.Y));
        if classnum==2
            file.Y(file.Y==0)=2;
        end
%         file.X=file.X(:,1:600);
%         file.X=file.X(1:800,:);
%         file.Y=file.Y(1:800,:);
        
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
        pars.classnum=classnum;%�����Ŀ

      
          pars.lambda1=[10^0];
          pars.lambda2=[10^1];
%         pars.lambda1 = [10^-3,10^-2,10^-1,10^0];
%         pars.lambda2 = [10^1,10^2,10^3];
        
        pars.r = min(m,n)*parm_r;
        pars.cvk = 1;  %������֤�Ĵ���   svmѵ������   
        pars.k = 10;
        pars.Ite = 20;
        pars.c = -5;
        pars.g = -2;
%         pars.c = -10:2:10;
%         pars.g = -10:2:10;
        pars.epsilon = 0.0001;
        libsvmType='liblinear';
        pars.libsvmType = libsvmType;
        
        pars.algorithm = char(algorithm(algorithmIndex));
    
        % % steps = 13; % should be changed by your code
%         for k = 1:pars.k
%             disp([num2str(k),'��']);
%             for labelIndex=1:classnum
%                 ind(:,k) = crossvalind('Kfold',size(find(label),1),pars.k);
%                 %ind(:,k) = crossvalind('Kfold',size(find(label == labelIndex),1),pars.k);
%             end
            %cross-validation
            test = (ind(:,k) == k); train = ~test;
            [testResults(k,:),mseResults(k,:)] = Step2_D_train_cv(X(train,:),X(test,:),label(train,:),label(test,:),pars,k,char(document(d))); 
%         end

        % close(h)
        meantestresult = mean(testResults)
        meanmseresutlt = mean(mseResults)
        if(strcmp(pars.algorithm,'xijAB_ABS'))
        	save([resultPath,char(document(d)),'_',pars.algorithm,'_',num2str(meantestresult),'% -',num2str(pars.lambda1(1)),'-',num2str(pars.lambda2(1)),'.mat']);
        else 
            save([resultPath,char(document(d)),'_',pars.algorithm,'_',num2str(meantestresult),'%.mat']);
        end
        %data ={'email','854289665@qq.com','subject',[char(document(d)),'_',pars.algorithm],'content',[char(document(d)),'_',pars.algorithm,'_',num2str(meantestresult)]};
        %urlread('http://172.16.25.68:8080/Mail/mail','POST',data);
        
        save('temp.mat','d','document','algorithmIndex','algorithm','datasetsPath','folderPath','datasetsPath','resultPath','parm_r');
        clear;
        load('temp.mat');
    end
    %%ѭ���Ա��㷨 end
    
    save('temp.mat','d','document','algorithmIndex','algorithm','datasetsPath','folderPath','datasetsPath','resultPath','parm_r');
    clear;
    load('temp.mat');
end
%%ѭ�����ݼ� end



