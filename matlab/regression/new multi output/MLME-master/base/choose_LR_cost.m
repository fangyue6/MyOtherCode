function [ cost_best ] = choose_LR_cost( X, Y, varargin )

Cost = [0.001 0.01 0.1 1 10 100];
k = 3;    %use 1/3 of the data for validation
indices = crossvalind('Kfold', length(Y), k);

if nargin > 2
    w = varargin{1};
end

for c=1:length(Cost)
    c_temp=Cost(c);
    paraLiblinear = ['-s 0 -B 1 -q -c ' num2str(c_temp)]; 

   
    LL_temp=[];
    acc_temp=[];
    for i=1:k
        index_validation = find(indices==i);
        index_train = find(indices~=i); 

        X_train=X(index_train,:);
        Y_train=Y(index_train);

        X_validation= X(index_validation,:);
        Y_validation = Y(index_validation);
        
%         if isequal(LR_implementation,'weighted_liblinear')
%             w_train=w(index_train);
%             w_validation = w(index_validation);
%             M = train(w_train, Y_train, sparse(double(X_train)),paraLiblinear);

        %M = train(Y_train, sparse(double(X_train)),paraLiblinear);
        M = train(ones(size(X_train,1),1), Y_train, sparse(double(X_train)),paraLiblinear);
        
        
        weights=[];
        weights(1) = M.w(end);
        weights = [weights M.w(1:end-1)];
        
        % flip the weights
        if(M.Label(1) == 0)
            weights = -weights;
        end
        
        [ P ] = LR_predict( weights, X_validation );
        Y_pred = P'>0.5;
        
        %[Y_pred, xx, P] = predict(Y_validation, sparse(double(X_validation)), M,'-b 1 -q');
       
        %classification accuracy
        acc_temp(i) = length(find(Y_validation==Y_pred))/length(Y_validation);
        LL_temp(i) = LR_likelihood(P,Y_validation);

    end
    acc(c) = mean(acc_temp);
    LL(c) = mean(LL_temp);
    
end

%[max_acc idx] =max(acc);
[max_LL idx] =max(LL);
cost_best=Cost(idx);
end

