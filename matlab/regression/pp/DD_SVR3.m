function [a,b,c,d,e,f,g,h,aa,bb,cc,dd,ee,ff,maxind] = DD_SVR3(PPDRXTrain,PPDRXTest,PCAXTrain,PCAXTest,Ytrain,Ytest,smartXTrain,smartXTest,CSFSXTrain,CSFSXTest,LSG21XTrain,LSG21XTest,SFUSXTrain,SFUSXTest,MSFSXTrain,MSFSXTest)

% function [a,b,aa,bb,ee,ff,maxind] = DD_SVR3(PPDRXTrain,PPDRXTest,Ytrain,Ytest,LSG21XTrain,LSG21XTest,MSFSXTrain,MSFSXTest)
%PPDR.aCC(j,i),PPDR.aRMSE(j,i),PCA1.aCC(j,i),PCA1.aRMSE(j,i),smart.aCC(j,i),smart.aRMSE(j,i),CSFS.aCC(j,i),CSFS.aRMSE(j,i),LSG21.aCC(j,i),LSG21.aRMSE(j,i),SFUS.aCC(j,i),SFUS.aRMSE(j,i),maxind(j,i)
%NewXTrain, NewXTest: ins*fea
%Ytrain,Ytest:ins*label

label = size(Ytrain,2);
parc = -5:1:5;
parg = -5:1:5;
% parc = -5;
% parg = -5;
for c = 1:1:length(parc)
    for g = 1:1:length(parg)
        cmd = ['-t 2 -s 4' ' -c ' num2str(2^parc(c)) ' -g ' num2str(10^parg(g))];   % regression parameter        
        
        for lab = 1:label
            Model_PPDR = svmtrain(Ytrain(:,lab),PPDRXTrain,cmd);
            [pre_PPDR, accuracy, dec] = svmpredict(Ytest(:,lab), PPDRXTest, Model_PPDR);
            PPDR_PreY(:,lab) = pre_PPDR; 
            
            Model_PCA = svmtrain(Ytrain(:,lab),PCAXTrain,cmd);
            [pre_PCA, accuracy, dec] = svmpredict(Ytest(:,lab), PCAXTest, Model_PCA);
            PCA_PreY(:,lab) = pre_PCA; 
            
            Model_smart = svmtrain(Ytrain(:,lab),smartXTrain,cmd);
            [pre_smart, smart_accuracy, smart_dec] = svmpredict(Ytest(:,lab), smartXTest, Model_smart);
            smart_PreY(:,lab) = pre_smart; 
%             
            Model_CSFS = svmtrain(Ytrain(:,lab),CSFSXTrain,cmd);
            [pre_CSFS, accuracy, dec] = svmpredict(Ytest(:,lab), CSFSXTest, Model_CSFS);
            CSFS_PreY(:,lab) = pre_CSFS;
            
            Model_LSG21 = svmtrain(Ytrain(:,lab),LSG21XTrain,cmd);
            [pre_LSG21, accuracy, dec] = svmpredict(Ytest(:,lab), LSG21XTest, Model_LSG21);
            LSG21_PreY(:,lab) = pre_LSG21; 
            
            Model_SFUS = svmtrain(Ytrain(:,lab),SFUSXTrain,cmd);
            [pre_SFUS, accuracy, dec] = svmpredict(Ytest(:,lab), SFUSXTest, Model_SFUS);
            SFUS_PreY(:,lab) = pre_SFUS; 
            
            Model_MSFS = svmtrain(Ytrain(:,lab),MSFSXTrain,cmd);
            [pre_MSFS, accuracy, dec] = svmpredict(Ytest(:,lab), MSFSXTest, Model_MSFS);
            MSFS_PreY(:,lab) = pre_MSFS
            
%             corr = corrcoef(pre_y(:,l),Ytest(:,l));
%             OES_corr(:,l) = corr(1,2);
%             OES_rmse(:,l) = rmse(pre_y(:,l),Ytest(:,l));
        end
        
        %SVR-PPDR
        PPDR_per = per_evl(Ytest,PPDR_PreY);     %等同于mean(OES_corr)和mean(OES_rmse)
        PPDR.aCC(c,g) = PPDR_per.aCC;
        PPDR.aRMSE(c,g) = PPDR_per.aRMSE;        
        
        %SVR-PCA
        PCA_per = per_evl(Ytest,PCA_PreY);     %等同于mean(OES_corr)和mean(OES_rmse)
        PCA.aCC(c,g) = PCA_per.aCC;
        PCA.aRMSE(c,g) = PCA_per.aRMSE;       
        
        %SVR-smart         
        smart_per = per_evl(Ytest,smart_PreY);     %等同于mean(OES_corr)和mean(OES_rmse)
        smart.aCC(c,g) = smart_per.aCC;
        smart.aRMSE(c,g) = smart_per.aRMSE;        
      
         %SVR-CSFS
        CSFS_per = per_evl(Ytest,CSFS_PreY);     %等同于mean(OES_corr)和mean(OES_rmse)
        CSFS.aCC(c,g) = CSFS_per.aCC;
        CSFS.aRMSE(c,g) = CSFS_per.aRMSE; 
        
        %SVR-LSG21
        LSG21_per = per_evl(Ytest,LSG21_PreY);     %等同于mean(OES_corr)和mean(OES_rmse)
        LSG21.aCC(c,g) = LSG21_per.aCC;
        LSG21.aRMSE(c,g) = LSG21_per.aRMSE;
        
         %SVR-SFUS
        SFUS_per = per_evl(Ytest,SFUS_PreY);     %等同于mean(OES_corr)和mean(OES_rmse)
        SFUS.aCC(c,g) = SFUS_per.aCC;
        SFUS.aRMSE(c,g) = SFUS_per.aRMSE;  
%         
         %SVR-MSFS
        MSFS_per = per_evl(Ytest,MSFS_PreY);     %等同于mean(OES_corr)和mean(OES_rmse)
        MSFS.aCC(c,g) = MSFS_per.aCC;
        MSFS.aRMSE(c,g) = MSFS_per.aRMSE;        
              
    end
end
[a maxind] = max(PPDR.aCC(:));
[b ~] = min(PPDR.aRMSE(:));

[c maxind] = max(PCA.aCC(:));
[d ~] = min(PCA.aRMSE(:));

[e maxind] = max(smart.aCC(:));
[f ~] = min(smart.aRMSE(:));

[g maxind] = max(CSFS.aCC(:));
[h ~] = min(CSFS.aRMSE(:));

[aa maxind] = max(LSG21.aCC(:));
[bb ~] = min(LSG21.aRMSE(:));

[cc maxind] = max(SFUS.aCC(:));
[dd ~] = min(SFUS.aRMSE(:));

[ee maxind] = max(MSFS.aCC(:));
[ff ~] = min(MSFS.aRMSE(:));
