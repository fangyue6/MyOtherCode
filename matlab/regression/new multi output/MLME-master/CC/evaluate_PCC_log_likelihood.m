function [ y_log_prob, yi_log_prob ] = evaluate_PCC_log_likelihood(M, x, y, P)
    d = size(M,2);
    for i=1:d
        temp_prob(P(i)) = LR_predict(M{i}, [x y(P(1:i-1))]);
    end
    [y_log_prob, ~, yi_log_prob] = compute_log_prob(y, temp_prob);
end
