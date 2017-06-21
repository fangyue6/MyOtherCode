%compute the likelihood of LR which learns from X to Y on the
%Crossvalidation splits in indices
function [ LL ] = compute_crossvalidation_loglikelihood( X, Y, indices, k)


%save time
learn_cost=false;

for i=1:k
    index_validation = find(indices==i);
    index_train = find(indices~=i); 

    Y_train=Y(index_train);
    Y_validation = Y(index_validation);

    %the model based on X only
    X_train=X(index_train,:);
    X_validation= X(index_validation,:);
    [ W1 ] = LR_train( X_train, Y_train, learn_cost );
    P=LR_predict( W1,X_validation );
    LL_temp(i) = LR_likelihood( P, Y_validation );
end

LL=mean(LL_temp);

end

