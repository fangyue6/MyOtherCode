%return P(y=1|x,w)
function [ P ] = LR_predict( w,X )
n=size(X,1);
P = [];

if size(w,1) == 1
    for i=1:n
        x=X(i,:);
        z=dot(w,[1 x]);
        P(i)=1.0 ./ (1.0 + exp(-z));
    end
else    % for multi-class
    
    for i=1:n
        for m = 1:size(w,1)
            x=X(i,:);
            z(m)=exp(dot(w(m,:),[1 x]));
        end
        
        for m = 1:size(w,1)
            P(i,m)=z(m)/sum(z);
        end
    end
end
