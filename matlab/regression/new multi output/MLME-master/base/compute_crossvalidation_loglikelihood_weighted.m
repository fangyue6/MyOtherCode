%compute the likelihood of LR which learns from X to Y on the
%Crossvalidation splits in indices
function [ LL ] = compute_crossvalidation_loglikelihood_weighted( X, Y, W, indices, k)

%save time
learn_cost=false;

%init
N = size(X,1);

for i=1:k
    index_validation = find(indices==i);
    index_train = find(indices~=i);

    Y_train = Y(index_train);
    Y_validation = Y(index_validation);
    
    W_train = N*W(index_train);
    W_validation = N*W(index_validation);
    

    %the model based on X only
    X_train=X(index_train,:);
    X_validation= X(index_validation,:);
    
    model = LR_train( X_train, Y_train, W_train, learn_cost );
    P = LR_predict( model, X_validation );
    LL_temp(i) = LR_likelihood_weighted( P, Y_validation, W_validation );
end

LL=mean(LL_temp);

end

