%learn_cost is a boolean variable, if false, do not optimize over for
%learning the cost, use the standard 1 cost (save time)
function [ weights, best_cost ] = LR_train( X, Y, varargin )

global LR_implementation;

cardinality = length(unique(Y));

is_learn_cost = true;
is_weighted = false;

% param setting
if nargin > 2
    for i=1:nargin-2
        if size(varargin{i},1) == size(X,1)
            w = varargin{i};
            is_weighted = true;
        elseif isequal(size(varargin{i}), [1 1])
            if islogical(varargin{i})
                is_learn_cost = varargin{i};
            else
                user_cost = varargin{i};
            end
        end
    end
end

if isequal(LR_implementation, 'liblinear')
    
    % learn cost
    if exist('user_cost', 'var')    % user_cost suppresses learn_cost
        best_cost = user_cost;
        
    elseif is_learn_cost
        %fprintf(2,'---- warn: learning cost ----\n');
        best_cost = choose_LR_cost( X, Y );
    else
        best_cost = 1;
    end
    
    % weight adjustment
%     if exist('w', 'var')
%         w = w/sum(w)*size(X,1);
%     end
    
    % learn parameters
    weights=[];
    if cardinality <= 2
        % binary instances
        param = [' -s 0 -B 1 -q -c ' num2str(best_cost)];
    
        if ~is_weighted
            % regular LR
            %M = train(Y, sparse(double(X)), param);
            M = train(ones(size(X,1),1), Y, sparse(double(X)), param);
        else
            % weighted LR
            M = train(w, Y, sparse(double(X)), param);
        end
        
        weights(1) = M.w(end);
        weights = [weights M.w(1:end-1)];
        
        % flip the weights
        if(M.Label(1) == 0)
            weights = -weights;
        end
        
    elseif cardinality > 2
        % multi-class instances
        
        param = ' -s 0 -B 1 -q -c 1';
        
        for i = 0:cardinality-1
            weights_temp = [];
            Y_binary = zeros(length(Y), 1);
            Y_binary(find(Y==i)) = 1;
            
            if ~is_weighted
                % regular LR
                %M = train(Y_binary, sparse(double(X)), param);
                M = train(ones(size(X,1),1), Y_binary, sparse(double(X)), param);
            else
                % weighted LR
                M = train(w, Y_binary, sparse(double(X)), param);
            end
                        
            weights_temp(1) = M.w(end);
            weights_temp = [weights_temp M.w(1:end-1)];
            
            % flip the weights
            if(M.Label(1) == 0)
                weights_temp = -weights_temp;
            end
            weights(i+1,:) = weights_temp;
        end
%         weights(:,1) = M.w(:,end);
%         weights=[weights M.w(:,1:end-1)];
%         
%         %check label order
%         for i = 1:length(M.Label)
%             j = find(M.Label == i-1);
%             w_temp(i,:) = weights(j,:);
%         end
%         weights=w_temp;
    end
    
else
    % (using Matlab built-in LR)
    weights = glmfit(X,[Y ones(length(Y),1)], 'binomial','link','logit');
end

end%end-of-function

