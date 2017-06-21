%LL: loglikelihood of [X,Y]
%avg_prob: the average probability for the true labels P(yi|xi)
function [ LL, avg_prob, prob ] = compute_loglikelihood( T, X, Y, is_switching )

% param set
if ~exist('is_switching', 'var');
    is_switching = true;
end

% init
LL = 0;
sum_prob = 0;
prob = [];
n = size(X, 1);

% proc
for i = 1:n
    
    Tx = compute_log_potentials(T, X(i,:), is_switching);
        
    ll(i) = evaluate_probability(Tx, Y(i,:));
    if(ll(i) < -100)
        ll(i) = -100;
    end
    
    prob(i,1) = exp(ll(i));
end

LL = sum(ll);
avg_prob = mean(exp(ll));
