function T = learn_weighted_structure( X, Y, W, prev_T, is_switching )

% param setting
if ~exist('is_switching', 'var');
    is_switching = true;
end
is_profiling = false;

% init
[N, d] = size(Y);
W = W/N;
% proc
if is_profiling, t1 = clock; end;

%3-folds cross validation
K=3;
%rand('seed',1);
indices = crossvalind('Kfold', N, K);
%load('idx_cv.mat');

idx=1;
S = [];
for i=1:d
    if mod(i, 10) == 0
        fprintf('...%d', i);
    end
    [ LL_without_Y ] = compute_crossvalidation_loglikelihood_weighted( X, Y(:,i), W, indices, K);
    node_weights(i) = LL_without_Y;
    for j = 1:d 

        if(j ~= i)
            % measure the influence of Y_j on Y_i
            if ~is_switching
                [ LL_with_y ] = compute_crossvalidation_loglikelihood_weighted( [X Y(:,j)], Y(:,i), W, indices, K);
            else
                [ LL_with_y ] = compute_crossvalidation_loglikelihood_sw_weighted( X, Y(:,i), Y(:,j), W, indices, K);
            end

            %condition
            if(LL_with_y > LL_without_Y)
                S{idx}.from = j;
                S{idx}.to = i;
                %S{idx}.weight = LL_with_y-LL_without_Y;
                S{idx}.weight = LL_with_y;
                idx = idx+1;
            end
        end

    end
end
 
if is_profiling, t2 = clock; end;

for i=1:d
    [T_list{i}, T_weight(i)] = directed_maximum_spanning_tree(S, 1:d, i, node_weights);
    
    if has( prev_T, T_list{i} )
        T_weight(i) = -inf;
    end
end

if sum(T_weight == -inf) == d
    while true
        T = generate_random_linear_chain(d);
        if ~has( prev_T, T )
            break;
        end
        % exception: if all possible structures are used
        if 2^(d-1) <= length(prev_T)
            fprintf(2,'warn: dupliate structure (running out of new structure)\n');
            break;
        end
    end
else
    [max_weight, max_idx] = max(T_weight);
    T = T_list{max_idx};
end

if is_profiling, t3 = clock; end;

[ T ] = organize_tree_BFS(T);

if is_profiling, t4 = clock; end;

if ~is_switching
    T = compute_tree_weights(T, X, Y);
else
    %disp('** opt: -s 1 -l 1');
    T = compute_tree_weights(T, X, Y, '-s 1 -l 1');
end

if is_profiling, t5 = clock; end;


if is_profiling,
    fprintf( '(pf)for-compute_edge_weight: %f s\n', etime(t2,t1) );
    fprintf( '(pf)directed_maximum_spanning_tree: %f s\n', etime(t3,t2) );
    fprintf( '(pf)organize_tree_BFS: %f s\n', etime(t4,t3) );
    fprintf( '(pf)compute_tree_weights: %f s\n', etime(t5,t4) );
end;

end


function is_duplicate = has(prev_T, T)

is_duplicate = true;

if isempty(prev_T)
    is_duplicate = false;
    return;
end

for t = 1:length(prev_T)
    is_match = true;
    for i = 1:length(prev_T{t})
        for j = 1:length(T)
            if prev_T{t}{i}.node == T{j}.node
                if ~isequal(prev_T{t}{i}.parent, T{j}.parent)
                    is_match = false;
                    break;
                elseif ~isequal(sort(prev_T{t}{i}.children), sort(T{j}.children))
                    is_match = false;
                    break;
                end
                break;
            end
        end
        if ~is_match
            break;
        end
    end
    if is_match
        return;
    end
end

is_duplicate = false;

end


function S = extract_structure(T)
for i = 1:length(T)
    S{i}.node = T{i}.node;
    S{i}.parent = T{i}.parent;
    S{i}.children = T{i}.children;
end
end



% function is_duplicate = has(prev_T, T)
% 
% is_duplicate = true;
% 
% if isempty(prev_T)
%     is_duplicate = false;
%     return;
% end
% 
% T = organize_tree_BFS(T);
% for i = 1:length(prev_T)
%     S = extract_structure(prev_T{i});
%     if isequal(S,T)
%         return;
%     end
% end
% 
% is_duplicate = false;
% 
% end
