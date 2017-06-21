%% CC_train: the classifier chain model [Read et al, 2009]
function [M, P, cost] = CC_train_weighted(X_tr, Y_tr, W_tr, P)

[~,d] = size(Y_tr);

if ~exist('P','var') || isempty(P)
    P = randperm(d);
end

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