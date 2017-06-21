%compute the likelihood of LR which learns from X to Y on the
%Crossvalidation splits in indices
function [ LL ] = compute_crossvalidation_loglikelihood_sw( X, Y, Y_parent, indices, k)


%save time
learn_cost=false;

for i=1:k
    index_validation = find(indices==i);
    index_train = find(indices~=i); 

    Y_train=Y(index_train);
    Y_validation = Y(index_validation);
    
    Y_parent_train = Y_parent(index_train);
    Y_parent_validation = Y_parent(index_validation);

    %the model based on X only
    X_train=X(index_train,:);
    X_validation= X(index_validation,:);
    
    
    no_instance0 = false;
    no_instance1 = false;
    
    roi = (Y_parent_train==0);
    if sum(roi) > 5
        [ W0 ] = LR_train( X_train(roi,:), Y_train(roi,:), learn_cost );
        roi0 = (Y_parent_validation==0);
        if sum(roi0) ~= 0
            P0 = LR_predict(W0,X_validation(roi0,:));
        end
    else
        no_instance0 = true;
    end
    roi = (Y_parent_train==1);
    if sum(roi) > 5
        [ W1 ] = LR_train( X_train(roi,:), Y_train(roi,:), learn_cost );
        roi1 = (Y_parent_validation==1);
        if sum(roi1) ~= 0
            P1 = LR_predict(W1,X_validation(roi1,:));
        end
    else
        no_instance1 = true;
    end
    
    if no_instance0    % W0 hasn't been trained -> use W1
        roi0 = (Y_parent_validation==0);
        if sum(roi0) ~= 0
            P0 = LR_predict(W1,X_validation(roi0,:));
        end
    elseif no_instance1    % W1 hasn't been trained -> use W0
        roi1 = (Y_parent_validation==1);
        if sum(roi1) ~= 0
            P1 = LR_predict(W0,X_validation(roi1,:));
        end
    end
    
    if sum(roi0) == 0
        LL_temp(i) = LR_likelihood( P1, Y_validation(roi1,:) );
    elseif sum(roi1) == 0
        LL_temp(i) = LR_likelihood( P0, Y_validation(roi0,:) );
    else
        LL_temp(i) = LR_likelihood( P0, Y_validation(roi0,:) ) + LR_likelihood( P1, Y_validation(roi1,:) );
    end
    
    % ---- add-on for multi-class case (start) ----
    if length(unique(Y_parent)) == 3
        no_instance2 = false;
        
        roi = (Y_parent_train==2);
        if sum(roi) ~= 0
            [ W2 ] = LR_train( X_train(roi,:), Y_train(roi,:), learn_cost );
            roi2 = (Y_parent_validation==2);
            if sum(roi2) ~= 0
                P2 = LR_predict(W2,X_validation(roi2,:));
            end
        else
            no_instance2 = true;
        end
        
        if sum(roi2) ~= 0
            LL_temp(i) = LL_temp(i) + LR_likelihood( P2, Y_validation(roi2,:) );
        end
        
        % uncertainty check
        if no_instance0 || no_instance1 || no_instance2
            % uncertainty arises: this scope shouldn't be triggered
            fprintf( 2, 'error: incorrect result is being produced! (compute_crossvalidation_loglikelihood_sw.m)\n' );
        end
    end
    % ---- add-on for multi-class case (end) ----
    
end

LL=mean(LL_temp);

end

