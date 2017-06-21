function [Y_pred, Y_log_prob, gate_xi] = MAP_prediction_MCTBN(base, gate, X, Y, is_switching)

if ~exist('is_switching','var')
    is_switching = true;
end

K = length(base);
[N,d] = size(Y);
m = size(X,2);
max_iter = 150;

if isempty(gate)
    gate = [zeros(K,m) ones(K,1)];
end

if d > 7
    [Y_pred, Y_log_prob, gate_xi] = MAP_prediction_MCTBN_SA(base, gate, X, Y, is_switching, max_iter);
else
    [Y_pred, Y_log_prob, gate_xi] = MAP_prediction_MCTBN_naive(base, gate, X, Y, is_switching );
end


end