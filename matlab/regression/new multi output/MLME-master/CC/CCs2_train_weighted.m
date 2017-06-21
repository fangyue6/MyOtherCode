%% CC_train: the classifier chain model [Read et al, 2009]
function [M, P, cost] = CCs2_train_weighted(X_tr, Y_tr, W_tr)

is_profiling = true;

[N, d] = size(Y_tr);


% learn (or generate) chain order

%P=perm(d);
%P=1:d;
P = zeros(1,d);

if is_profiling, t1 = clock; end;

% 3-folds internal cross validation
k = 3;
indices = crossvalind('Kfold', Y_tr(:,1), k);

for j = 1:d
    if mod(j, 10) == 0, fprintf('...%d', j);end;
    LL = -inf(1,d);
    for i = 1:d
        if sum(P == i) == 0
            %    LL(i) = compute_crossvalidation_loglikelihood( X_tr, Y_tr(:,i), indices, k);
            LL(i) = compute_crossvalidation_loglikelihood_weighted( [X_tr Y_tr(:,P(1:j-1))], Y_tr(:,i), W_tr, indices, k);
        else
            LL(i) = nan;
        end
    end
    [maxLL, maxLL_i] = max(LL);
    if isnan(maxLL)
        fprintf(2,'error: maxLL looks strange (nan).\n');
    elseif isinf(maxLL)
        fprintf('warn: maxLL looks strange (-inf).\n');
    end
    P(j) = maxLL_i;
end

if is_profiling, t2 = clock; end;


% train model
M = cell(1, d);
cost = zeros(1, d);
for i = 1:d
    %M{i} = LR_train([X_tr Y_tr(:,1:i-1)], Y_tr(:,i));
    % hard-wiring
    Yb_tr_prev = [];
    if i > 1
        Yb_tr_prev = convert_m2b(Y_tr(:,P(1:i-1)));
    end
    [M{i}, cost(i)] = LR_train([X_tr Yb_tr_prev], Y_tr(:,P(i)), W_tr);
end

end %end of function CC_train()