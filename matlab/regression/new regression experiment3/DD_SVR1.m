function [a,b,maxind] = DD_SVR1(Ytrain,Ytest,smartXTrain,smartXTest)
%PPDRXTrain, PPDRXTest, Y_tr, Y_te
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
           
            Model_smart = svmtrain(Ytrain(:,lab),smartXTrain,cmd);
            [pre_smart, smart_accuracy, smart_dec] = svmpredict(Ytest(:,lab), smartXTest, Model_smart);
            smart_PreY(:,lab) = pre_smart; 

        end 
        
        %SVR-smart         
        smart_per = per_evl(Ytest,smart_PreY);     %等同于mean(OES_corr)和mean(OES_rmse)
        smart.aCC(c,g) = smart_per.aCC;
        smart.aRMSE(c,g) = smart_per.aRMSE;                
        
%         [Multioutput.aCC(c,g) , Multioutput.aRMSE(c,g)]= Multioutput_regression_evaluation(Ytest,smart_PreY);     %等同于mean(OES_corr)和mean(OES_rmse)
       
%         [smart.aCC smart.aRMSE] = Multioutput_regression_evaluation(Ytest,smart_PreY);
    end
end

[a maxind] = max(smart.aCC(:));
[b ~] = min(smart.aRMSE(:));
