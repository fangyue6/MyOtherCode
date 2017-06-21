function [Y_pred, Y_log_prob, gate_xi] = MAP_prediction_MCC_naive( CC, gate, X, Y )

K = length(CC);
[N,d] = size(Y);
m = size(X,2);
Y_log_prob = zeros(N,1);
Y_pred = zeros(N,d);
gate_xi = zeros(N,K);

if isempty(gate)
    gate = [zeros(K,m) ones(K,1)];
end

Y_comb = generate_all_bin_combinations(d);

for i=1:N
    x = X(i,:);
    exp_theta_x = zeros(1,K);
    Y_comb_prob = zeros( size(Y_comb,1), 1 );
    
    for k = 1:K
        exp_theta_x(k) = exp(dot( gate(k,:), [x 1] ));
    end
    g_xi = exp_theta_x / sum(exp_theta_x);
    
    for yi = 1:size(Y_comb,1)
        y = Y_comb(yi,:);
        
        tmp_prob = zeros(1,K);
        for k = 1:K
            %[~,tmp_prob(k)] = CC_predict(CC{k}.model, x, y, CC{k}.permutation);
            tmp_prob(k) = exp(evaluate_PCC_log_likelihood(CC{k}.model, x, y, CC{k}.permutation));
        end
        Y_comb_prob(yi) = dot(tmp_prob, g_xi);
    end
    
    [~,Y_map_i] = max(Y_comb_prob);
    Y_pred(i,:) = Y_comb(Y_map_i,:);
    
    % Y_log_prob
    for k = 1:K
    	tmp_prob(k) = exp(evaluate_PCC_log_likelihood(CC{k}.model, X(i,:), Y(i,:), CC{k}.permutation));
    end
    Y_log_prob(i,1) = log(dot(tmp_prob, g_xi));
    gate_xi(i,:) = g_xi;
end

% fprintf( '.' );

end


