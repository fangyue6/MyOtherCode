%Y: true labels, Y_pred: predicted labels, Y_log_prob: log probability of the true class
%according to the model
function [ obj ] = getMeasuresMLC( Y, Y_pred, Y_log_prob )

    
[N d] = size(Y);


%% Compute Exact Matching rate and Hamming
cnt = 0;
cnt2 = 0;

for i=1:N
    y = Y(i,:);
    y_pred = Y_pred(i,:);
    if (isequal(y,y_pred))
        cnt = cnt +1;
    end
    cnt2 = cnt2 + sum(y==y_pred);
end

obj.ExactMatch = cnt/N; 
obj.HammingMatch=cnt2/(N*d);

cnt3 = 0;
for i=1:N
    y = Y(i,:);
    y_pred = Y_pred(i,:);
    
    if sum(y | y_pred) ~= 0
        cnt3 = cnt3 + sum(y & y_pred)/sum(y | y_pred);
    else
        cnt3 = cnt3 + 1;
    end
end
obj.MLAccuracy = cnt3/N; 

%% Compute Macro Average F1: take the average over all labels
for j=1:d
    if(length(find(Y_pred(:,j)==1))==0)
        P(j)=1;
    else        
        P(j)=length(find(Y(:,j)==1 & Y_pred(:,j)==1))/length(find(Y_pred(:,j)==1));
    end

    if(length(find(Y(:,j)==1))==0)
        R(j)=1;
    else
        R(j)=length(find(Y(:,j)==1 & Y_pred(:,j)==1))/length(find(Y(:,j)==1));
    end

end

macro_P=mean(P);
macro_R=mean(R);

if(macro_P==0 & macro_R==0)
    macro_F=0;
else
    macro_F=(2*macro_P*macro_R)/(macro_P+macro_R);
end

obj.MacroF1=macro_F;

   
   
    
%% Compute Micro Average F1
% Micro-averaged F-measures: aggregate true positives, true negatives and  
% false positives and false negatives over labels, and then calculate an
% F-measure from them

TP=0;
P_pred=0;
P_true=0;

for j=1:d
    TP=TP+length(find(Y(:,j)==1 & Y_pred(:,j)==1));
    P_pred=P_pred+length(find(Y_pred(:,j)==1));
    P_true=P_true+length(find(Y(:,j)==1));
end

if(P_pred==0)
    micro_P=1;
else
    micro_P=TP/P_pred;
end

micro_R=TP/P_true;


if(micro_P==0 & micro_R==0)
    micro_F=0;
else
    micro_F=(2*micro_P*micro_R)/(micro_P+micro_R);
end

obj.MicroF1=micro_F;
% 
% %% Compute Pairewise Match
% 
% for i=1:numExamples
%     pMatch(i) = getPairwiseMatch(Y(i,:),Y_hat_binary(i,:));
% end
% obj.PairewiseMatch = mean(pMatch);
%    



%if the method is probabilistic, compute loglikelihood
if(nargin>2)
    obj.ll= sum(Y_log_prob);
end
