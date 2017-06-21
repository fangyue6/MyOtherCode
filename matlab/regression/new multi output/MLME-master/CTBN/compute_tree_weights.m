function [ T ] = compute_tree_weights( T, X, Y, varargin )

is_learn_cost = true;
is_weighted = false;
is_switching = false;

param=[];
user_cost=[];
N = size(X,1);
best_cost = [];

% param setting
if nargin == 4
    if size(varargin{1},1) == 1
        param = varargin{1};
    elseif size(varargin{1},1) == size(X,1)
        W = varargin{1};
    end
elseif nargin > 4
    for i=1:nargin-3
        if size(varargin{i},1) == 1
            param = varargin{i};
        elseif size(varargin{i},1) == size(X,1)
            W = varargin{i};
        end
    end
end

while ~isempty(param)
    [lp, param] = strtok(param);
    param = strtrim(param);
    if isempty(param), fprintf('error: incorrect parameters\n'); return; end
    [rp, param] = strtok(param);
    param = strtrim(param);
    
    if strcmp(lp, '-l')
        if strcmp(rp, '1') || strcmp(rp, 'true')
            is_learn_cost = true;
        elseif strcmp(rp, '0') || strcmp(rp, 'false')
            is_learn_cost = false;
        else
            fprintf('error: incorrect parameter value\n');
            return;
        end
    elseif strcmp(lp, '-c')
        user_cost = str2num(rp);
        if isempty(rp)
            fprintf('error: incorrect parameter value\n');
            return;
        end
    elseif strcmp(lp, '-s')
        if strcmp(rp, '1') || strcmp(rp, 'true')
            is_switching = true;
        elseif strcmp(rp, '0') || strcmp(rp, 'false')
            is_switching = false;
        else
            fprintf('error: incorrect parameter value\n');
            return;
        end
    else
        fprintf('error: unknown parameters\n');
        return;
    end
end

if exist('w','var')
    is_weighted = true;
end

if ~isempty(user_cost) && is_learn_cost
    fprintf('warn: learn_cost is forced to be ''false'' because cost is manually given\n');
    is_learn_cost = false;
    best_cost = user_cost;
end


% compute tree weights
for i=1:length(T)
    
    if mod(i,100) == 0
        fprintf('...%d', i);
    end
    
    node = T{i}.node;
    T{i}.card = max(2,length(unique(Y(:,T{i}.node))));
    T{i}.weights = [];
    
    
    
    if(isempty(T{i}.parent))
        % independent node
        if ~is_weighted
            % regular LR
            if ~isempty(user_cost)   % cost in user param has higher priority
                T{i}.weights = LR_train(X, Y(:,node), false, user_cost);
            elseif isfield(T{i}, 'best_cost')
                T{i}.weights = LR_train(X, Y(:,node), false, T{i}.best_cost);
            else
                [T{i}.weights, best_cost] = LR_train(X, Y(:,node), is_learn_cost);
            end
        else
            % weighted LR
            if ~isempty(user_cost)   % cost in user param has higher priority
                T{i}.weights = LR_train(X, Y(:,node), W*N, false, user_cost);
            elseif isfield(T{i}, 'best_cost')
                T{i}.weights = LR_train(X, Y(:,node), W*N, false, T{i}.best_cost);
            else
                [T{i}.weights, best_cost] = LR_train(X, Y(:,node), W*N, is_learn_cost);
            end
        end
        
    else
        % dependent node
        parent=T{i}.parent;
        
        if is_switching
            % switching model (feature is NOT enriched by adding Y(parent))
            no_instance = false(1,max(2,T{i}.card));
            is_learned_best_cost = false(1,T{i}.card);
            
            for j = 1:T{i}.card
                % case: [parent==(j-1)]
                roi = (Y(:,parent)==(j-1));
                if sum(roi) > 15
                    if ~is_weighted
                        % regular LR
                        if ~isempty(user_cost)   % cost in user param has higher priority
                            T{i}.weights{j} = LR_train(X(roi,:), Y(roi,node), false, user_cost);
                        elseif isfield(T{i}, 'best_cost') && ~isnan(T{i}.best_cost(j))
                            T{i}.weights{j} = LR_train(X(roi,:), Y(roi,node), false, T{i}.best_cost(j));
                        else
                            [T{i}.weights{j}, best_cost(j)] = LR_train(X(roi,:), Y(roi,node), is_learn_cost);
                            is_learned_best_cost(j) = true;
                        end
                    else
                        % weighted LR
                        if ~isempty(user_cost)   % cost in user param has higher priority
                            T{i}.weights{j} = LR_train(X(roi,:), Y(roi,node), W(roi)*N, false, user_cost);
                        elseif isfield(T{i}, 'best_cost') && ~isnan(T{i}.best_cost(j))
                            T{i}.weights{j} = LR_train(X(roi,:), Y(roi,node), W(roi)*N, false, T{i}.best_cost(j));
                        else
                            [T{i}.weights{j}, best_cost(j)] = LR_train(X(roi,:), Y(roi,node), W(roi)*N, is_learn_cost);
                            is_learned_best_cost(j) = true;
                        end
                    end
                else
                    no_instance(j) = true;
                end
            end
            
            % uncertainty handling (only works for binary case)
            if no_instance(1)
                T{i}.weights(1,:) = T{i}.weights(2,:);
            elseif no_instance(2)
                T{i}.weights(2,:) = T{i}.weights(1,:);
            end
            
            if sum(no_instance == true) > 0
                fprintf( 2, 'error: incorrect result is being produced! (compute_tree_weights_sw.m)\n' );
            end

        else
            % monolithic model
            if ~is_weighted
                % regular LR
                if ~isempty(user_cost)   % cost in user param has higher priority
                    T{i}.weights = LR_train([X Y(:,parent)], Y(:,node), false, user_cost);
                elseif isfield(T{i}, 'best_cost')
                    T{i}.weights = LR_train([X Y(:,parent)], Y(:,node), false, T{i}.best_cost);
                else
                    [T{i}.weights, best_cost] = LR_train([X Y(:,parent)], Y(:,node), is_learn_cost);
                end
            else
                % weighted LR
                if ~isempty(user_cost)   % cost in user param has higher priority
                    T{i}.weights = LR_train([X Y(:,parent)], Y(:,node), W*N, false, user_cost);
                elseif isfield(T{i}, 'best_cost')
                    T{i}.weights = LR_train([X Y(:,parent)], Y(:,node), W*N, false, T{i}.best_cost);
                else
                    [T{i}.weights, best_cost] = LR_train([X Y(:,parent)], Y(:,node), W*N, is_learn_cost);
                end
            end
        end %end of if(is_switching)
    end %end of if(isempty(T{i}.parent))
    
    % save learned_best_cost in structure T
    if is_learn_cost && ~isempty(best_cost)
        if ~is_switching || isempty(T{i}.parent)
            T{i}.best_cost = best_cost;
        else
            for j = 1:T{i}.card
                if is_learned_best_cost(j)
                    T{i}.best_cost(j) = best_cost(j);
                else
                    T{i}.best_cost(j) = NaN;
                end
            end
        end
    end
    
end%end of for i=1:length(T)


end %end of function compute_tree_weights_sw()


