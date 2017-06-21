function [Y_pred, Y_log_prob, gate_xi] = MAP_prediction_MCC(base, gate, X, Y)

K = length(base);
[N,d] = size(Y);
m = size(X,2);
max_iter = 150;

if isempty(gate)
    gate = [zeros(K,m) ones(K,1)];
end

if d > 7
    [Y_pred, Y_log_prob, gate_xi] = MAP_prediction_MCC_SA(base, gate, X, Y, max_iter);
else
    [Y_pred, Y_log_prob, gate_xi] = MAP_prediction_MCC_naive(base, gate, X, Y);
end


end