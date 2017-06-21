%compute the likelihood of Y under probabilities P
%P(i) is Prob(Y(i)=1)
function [ LL ] = LR_likelihood_weighted( P, Y, W )

MIN_LL = -100;
[r,c] = size(P);

if min(r,c) == 1 || min(r,c) == 2
    
    %LogLikelihood
    LL=0;
    n=length(Y);
    for i=1:n
        if(Y(i)==1)
            if P(i) <= 0
                LL=LL+W(i)*MIN_LL;
            else
                LL=LL+W(i)*log(P(i));
            end
        else
            if 1-P(i) <= 0
                LL=LL+W(i)*MIN_LL;
            else
                LL=LL+W(i)*log(1-P(i));
            end
        end
    end
    
else    %multi-class
    
    %LogLikelihood
    LL=0;
    n=length(Y);
    for i=1:n
        if P(i,Y(i)+1) <= 0
            LL=LL+W(i)*MIN_LL;
        else
            LL=LL+W(i)*log(P(i,Y(i)+1));
        end
    end
    
end



% if(Y(i)==1)
%     LL=LL+log(P(i,1));
% elseif(Y(i)==2)
%     LL=LL+log(P(i,2));
% else
%     LL=LL+log(P(i,3));
% end