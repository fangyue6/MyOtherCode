function ME = train_MLME(X, Y, K_min, gate_lambda, n_iter, BASE_MLC, is_switching)

% if(runParallel && matlabpool('size')==0)
%     matlabpool('open')
% end

if ~exist('is_switching','var')
    is_switching = true;
end

[N, d] = size(Y);
m = size(X, 2);

% for internal cross validation
N_icv_tr = floor(N*0.7);
idx_randperm = randperm(N)';
X_icv_tr = X(idx_randperm(1:N_icv_tr),:);
Y_icv_tr = Y(idx_randperm(1:N_icv_tr),:);
X_icv_val = X(idx_randperm(N_icv_tr+1:end),:);
Y_icv_val = Y(idx_randperm(N_icv_tr+1:end),:);

base = [];
W_icv_tr = [];

% learn the first base MLC model
k = 1;
gate = [];
W_icv_tr{k} = ones(N, 1);
if strcmp(BASE_MLC, 'CC')
    [base{k}.model, base{k}.permutation, base{k}.cost] = CCs2_train_weighted(X_icv_tr, Y_icv_tr, W_icv_tr{k});
    [~, Y_log_prob_val] = MAP_prediction_MCC(base, gate, X_icv_val, Y_icv_val);
elseif strcmp(BASE_MLC, 'CTBN')
    base{k} = learn_weighted_structure(X_icv_tr, Y_icv_tr, W_icv_tr{1}, [], is_switching);
    [~,~,Y_log_prob_val] = compute_loglikelihood(base{1}, X_icv_val, Y_icv_val, is_switching);
    gate = [zeros(k,m) ones(k,1)/k];
end

prev_log_prob_val = sum(Y_log_prob_val);
prev_base = base;
prev_gate = gate;

% add additional base MLC models
for k = 2:(3*K_min)
    fprintf(':: k = %d ::\n', k);
    
    % calc instance weights
    if strcmp(BASE_MLC, 'CC')
        [~, Y_log_prob_train] = MAP_prediction_MCC(base, gate, X_icv_tr, Y_icv_tr);
    elseif strcmp(BASE_MLC, 'CTBN')
        [~,~,Y_log_prob_train] = MCTBN_compute_loglikelihood(base, gate, X_icv_tr, Y_icv_tr, is_switching);
    end
    W_icv_tr{k} = 1 - exp(Y_log_prob_train);
    W_icv_tr{k} = W_icv_tr{k} / sum(W_icv_tr{k}) * N_icv_tr;
    
    if strcmp(BASE_MLC, 'CC')
        [base{k}.model, base{k}.permutation, base{k}.cost] = CCs2_train_weighted(X_icv_tr, Y_icv_tr, W_icv_tr{k});
        [base, gate] = MCC_learn_ME(base, X_icv_tr, Y_icv_tr, [], [], n_iter, gate_lambda);
        [~, Y_log_prob_val] = MAP_prediction_MCC(base, gate, X_icv_val, Y_icv_val);
    elseif strcmp(BASE_MLC, 'CTBN')
        base{k} = learn_weighted_structure(X_icv_tr, Y_icv_tr, W_icv_tr{k}, [], is_switching);
        [base, gate] = MCTBN_learn_ME(X_icv_tr, Y_icv_tr, base, is_switching, [], [], n_iter, gate_lambda);
        [~,~,Y_log_prob_val] = MCTBN_compute_loglikelihood(base, gate, X_icv_val, Y_icv_val, is_switching);
    end
    
    % sentinel: the mixture is saturated
    if prev_log_prob_val >= sum(Y_log_prob_val) && k >= K_min
        base = prev_base;
        gate = prev_gate;
        
        if strcmp(BASE_MLC, 'CC')
            [base, gate] = MCC_learn_ME(base, X, Y, [], [], n_iter*3, gate_lambda);
        elseif strcmp(BASE_MLC, 'CTBN')
            [base, gate] = MCTBN_learn_ME(X, Y, base, is_switching, [], [], n_iter*3, gate_lambda);
        end
        
        break;
    end
    % otherwise
    prev_log_prob_val = sum(Y_log_prob_val);
    prev_base = base;
    prev_gate = gate;
end

ME = [];
ME.base = base;
ME.gate = gate;

% if(runParallel && matlabpool('size')~=0)
%     matlabpool('close')
% end

end
