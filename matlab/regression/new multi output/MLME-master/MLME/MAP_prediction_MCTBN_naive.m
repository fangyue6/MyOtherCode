function [Y_pred, Y_log_prob, gate_xi] = MAP_prediction_MCTBN_naive( Experts, gate, X, Y, is_switching )

if ~exist('is_switching','var')
    is_switching = true;
end

K = length(Experts);
d = length(Experts{1});
n = size(X,1);
Y_log_prob = zeros(n, 1);
Y_pred = zeros(n, d);
gate_xi = zeros(n, K);

Y_comb = generate_all_bin_combinations(d);

for i=1:n
% 	if mod(i,10) == 1
% 		fprintf( '.' );
%     end
    
    Y_comb_prob = zeros( size(Y_comb,1), 1 );
    
    x=X(i,:);
    exp_theta_x = zeros(1,K);
    
    Tx = cell(1,K);
    for k = 1:K
        Tx{k} = compute_log_potentials( Experts{k}, x, is_switching );
        exp_theta_x(k) = exp(dot( gate(k,:), [x 1] ));
    end
    
    g_xi = exp_theta_x / sum(exp_theta_x);
    
    
    % goes over all possible combinations of Y
    for yi = 1:size(Y_comb,1)
        y = Y_comb(yi,:);
        tmp_prob = zeros(1,K);
        for k = 1:K
            tmp_prob(k) = exp(evaluate_probability( Tx{k}, y ));
        end
        Y_comb_prob(yi) = dot(tmp_prob, g_xi);
        
    end
    
    [~,Y_MAP_i] = max(Y_comb_prob);
    Y_pred(i,:) = Y_comb(Y_MAP_i,:);
    
    % Y_log_prob : P(Y_true)
    tmp_prob = zeros(1,K);
    for k = 1:K
    	tmp_prob(k) = exp(evaluate_probability( Tx{k}, Y(i,:) ));
    end
    Y_log_prob(i,1) = log(dot(tmp_prob, g_xi));
    gate_xi(i,:) = g_xi;
end

% fprintf( '.' );

end