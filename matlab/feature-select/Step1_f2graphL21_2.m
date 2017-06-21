% function Demo_f2L21(dataset,libsvmType)
%%%%%%%%%%%% NOTE THAT %%%%%%%%%%%%%%%%%%%
close all;
clc;clear;
%  before runing the demo, you should have installed LIBSVM and change  the following  path:
 addpath(genpath('F:\Program Files\MATLAB\MATLAB Production Server\R2015a\toolbox\libsvm-3.20\matlab'));
% 
% 


traindocument={'train_RSR_76.0294%','train_jelsr_67.1691%','train_CSFS_79.8529%','train_FSASL_74.3382%','train_FSSI_71.4338%','train_RPCA_OM_79.2647%','train_xijAB_ABS_84.9265%'};
DBWorlddocument={'DBWorld-30%-R90%-700_RSR_51.6667%','DBWorld-30%-R90%-700_jelsr_73.3333%','DBWorld-30%-R90%-700_CSFS_80%','DBWorld-30%-R90%-700_FSASL_81.6667%','DBWorld_FSSI_66.9048%','DBWorld_RPCA_OM_68.0952%','DBWorld-30%-R90%-700_xijAB_ABS_84.5238%'};
ALLAMLdocument={'ALLAML_RSR_88.75%','ALLAML_jelsr_72.1429%','ALLAML_CSFS_90.8929%','ALLAML_FSASL_63.5714%','ALLAML_FSSI_83.0357%','ALLAML_RPCA_OM_82.1429%','ALLAML_xijAB_ABS_95.7143%'};
PCMACdocument={'PCMAC_RSR_60.6429%','PCMAC_jelsr_51.5456%','PCMAC_CSFS_77.5971%','PCMAC_FSASL_73.7312%','PCMAC_FSSI_59.7143%','PCMAC_RPCA_OM_73.3317%','PCMAC_xijAB_ABS_79.898% -0.1-10'};

% Umistdocument={'umist_RSR_98.43','umist_jelsr_93.55','umist_CSFS_97.2293%','umist_FSASL_98.4392%','umist_FSSI_93.7387%','umist_RPCA_OM_98.4301%','umist_xijAB_ABS_99.6491%'};
Yaledocument={'Yale15_RSR_74.8162%','Yale15_jelsr_70.2941%','Yale15_CSFS_69.7059%','Yale15_FSASL_72.7941%','Yale15_FSSI_75.1103%','Yale15_RPCA_OM_69.0441%','Yale15_xijAB_ABS_81.2868%'};
ORLdocument={'ORL40_RSR_87.75%','ORL40_jelsr_95.75%','ORL40_CSFS_90.5%','ORL40_FSASL_90.25%','ORLj40_FSSI_94.5%','ORL40_RPCA_OM_93%','ORL40_xijAB_ABS_97% -0.001-10'};
pixraw10Pdocument={'pixraw10P_RSR_80%','pixraw10P_jelsr_73%','pixraw10P_CSFS_72%','pixraw10P_FSASL_78%','pixraw10P_FSSI_73%','pixraw10P_RPCA_OM_89%','pixraw10P_xijAB_ABS_97%'};
lungdocument={'lung_RSR_79.1667%','lung_jelsr_81.5476%','lung_CSFS_79.0952%','lung_FSASL_70.619%','lung_FSSI_92.6429%','lung_RPCA_OM_92.1667%','lung_xijAB_ABS_95.5952%'}
gli85document={'GLI-85_RSR_85.4167%','GLI-85_jelsr_68.6111%','GLI-85_CSFS_87.5%','GLI-85_FSASL_72.3611%','GLI-85_FSSI_88.0556%','GLI-85_RPCA_OM_90.5556%','GLI-85_xijAB_ABS_94.1667%'};


%% 定义参数start
folderPath='D:\fangyue\algorithm\feature-select\';
datasetsPath=[folderPath,'datasets\'];
resultPath=[folderPath,'result\compare1\'];
%'DBWorld-30%-R90%-700'  'ALLAML' 'PCMAC' 'GLI-85' 'train' 'Yale15' 'ORL40' 'pixraw10P' 'lung'
% 'RSR' 'jelsr' 'CSFS' 'FSASL' 'FSSI' 'RPCA_OM' 'xijAB_ABS'
document = {'Yale15'};
algorithm ={'RSR' 'jelsr' 'CSFS' 'FSASL' 'FSSI' 'RPCA_OM' 'xijAB_ABS'};
parm_r=1;
myDocument=Yaledocument
%%定义参数end

%% 循环数据集 start
for d = 1:length(document)
    
    %% 循环对比算法 start
    for algorithmIndex = 4:length(algorithm)
        
        fileName = [datasetsPath,char(document(d)) '.mat'];
        file = load(fileName);
        
        myResult=load([folderPath,'result\compare\利用\',char(myDocument(algorithmIndex)),'.mat']);

        %与数据对应的类数
%         classnum = length(unique(file.Y));
%         if classnum==2
%             file.Y(file.Y==0)=2;
%         end
          classnum  = myResult.classnum;
%         file.X=file.X(:,1:600);
%         file.X=file.X(1:800,:);
%         file.Y=file.Y(1:800,:);
        
        [m n]=size(file.X);
%         X=full(file.X);
%         
%         %X=selectX(X,0.5);
%         %  normalize each row to unit
%         %X = X./repmat(sqrt(sum(X.^0,2)),1,size(X,2));
%         %  normalize each column to unit
%         %X = X./repmat(sqrt(sum(X.^2,1)),size(X,1),1);
%         X = NormalizeFea(X,0);
        X=myResult.X
%         clear ans info
%         label=file.Y;
        label=myResult.label;

        pars = [];
        pars.classnum=classnum;%类别数目

      
          pars.lambda1=myResult.pars.lambda1;
          pars.lambda2=myResult.pars.lambda2;
%         pars.lambda1 = [10^-3];
%         pars.lambda2 = [10^1];
        
        pars.r = myResult.pars.r;
        pars.cvk =  myResult.pars.cvk;  %交叉验证的次数   svm训练参数   
        pars.k = 10;
        pars.Ite = 20;
        pars.c =myResult.pars.c;
        pars.g = myResult.pars.g;
%         pars.c = -10:2:10;
%         pars.g = -10:2:10;
        pars.epsilon = 0.0001;
        libsvmType='liblinear';
        pars.libsvmType = libsvmType;
        
        pars.algorithm = char(algorithm(algorithmIndex));
    
        % % steps = 13; % should be changed by your code
        for k = 1:pars.k
            disp([num2str(k),'折']);
            for labelIndex=1:classnum
                ind(:,k) = crossvalind('Kfold',size(find(label),1),pars.k);
                %ind(:,k) = crossvalind('Kfold',size(find(label == labelIndex),1),pars.k);
            end
            %cross-validation
%             ind=myResult.ind;
            test = (ind(:,k) == k); train = ~test;
            
%             compute_para.Xtrain(k,:)=X(train,:);
%             compute_para.Xtest(k,:)=X(test,:);
%             compute_para.labeltrain(k,:)=label(train,:);
%             compute_para.labeltest(k,:)=label(test,:);
            [opt,pred_label,testResults(k,:),mseResults(k,:)] = Step2_D_train_cv(X(train,:),X(test,:),label(train,:),label(test,:),pars,k,char(document(d))); 
            eval(['opt_',num2str(k),'=opt'])
            eval(['pred_label_',num2str(k),'=pred_label'])
            [FPR.F(k,:),FPR.P(k,:),FPR.R(k,:)]=compute_f(pred_label,label(test,:));
            
        end
          
        % close(h)
        meantestresult = mean(testResults)
        meanmseresutlt = mean(mseResults)
        meanF = mean(FPR.F)
        meanP = mean(FPR.P)
        meanR = mean(FPR.R)
        if(strcmp(pars.algorithm,'xijAB_ABS'))
        	save([resultPath,char(document(d)),'_',pars.algorithm,'_',num2str(meantestresult),'%.mat']);
        else 
            save([resultPath,char(document(d)),'_',pars.algorithm,'_',num2str(meantestresult),'%.mat']);
        end
        %data ={'email','854289665@qq.com','subject',[char(document(d)),'_',pars.algorithm],'content',[char(document(d)),'_',pars.algorithm,'_',num2str(meantestresult)]};
        %urlread('http://172.16.25.68:8080/Mail/mail','POST',data);

    end
    %%循环对比算法 end
    
end
%%循环数据集 end



