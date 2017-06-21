%LL: loglikelihood of [X,Y]
%avg_prob: the average probability for the true labels P(yi|xi)
function [ LL, avg_prob, Y_log_prob ] = MCTBN_compute_loglikelihood( Experts, gate, X, Y, is_switching )

global runParallel;
global num_batches;

if ~exist(num_batches, 'var') || ~exist(num_batches, 'var')
    is_parallelized = false;
elseif runParallel && num_batches <= size(X,1)
    is_parallelized = true;
else
    is_parallelized = false;
end

if is_parallelized
    [ LL, avg_prob, Y_prob ] = compute_loglikelihood_parallel( Experts, gate, X, Y, is_switching );
else
    [ LL, avg_prob, Y_prob ] = compute_loglikelihood_sequential( Experts, gate, X, Y, is_switching );
end

Y_log_prob = log(Y_prob);

end%end-of-function


%%%%
function [ LL, avg_prob, Y_prob ] = compute_loglikelihood_sequential( Experts, gate, X, Y, is_switching )

K = length(Experts);
n = size(X,1);
Y_prob = zeros(n,1);


for i = 1:n
    tmp_prob = zeros(1,K);
    exp_theta_x = zeros(1,K);
    
    for k = 1:K
        [~, tmp_prob(k)] = compute_loglikelihood( Experts{k}, X(i,:), Y(i,:), is_switching );
        %tmp_prob(k) = exp(evaluate_probability( Trees{k}, Y(i,:)) );
        exp_theta_x(k) = exp(dot( gate(k,:), [X(i,:) 1] ));
    end
    
    g_xi = exp_theta_x / sum(exp_theta_x);
    Y_prob(i,1) = dot(tmp_prob, g_xi);
end

LL = sum(log(Y_prob));
avg_prob = mean(Y_prob);

end%end-of-function compute_loglikelihood_MT_sequential()


%%%%
function [ LL, avg_prob, Y_prob ] = compute_loglikelihood_parallel( Experts, gate, X, Y, is_switching )

global num_batches;

%K = length(Experts);
n = size(X,1);
%prob = zeros(n,1);
%exp_theta_x = zeros(1,K);

batch_size = round(n/num_batches);
batch_size = repmat(batch_size, [1 num_batches]);

s = 1;
e = s+batch_size(1)-1;
for i = 1:num_batches
    X_temp{i} = X(s:e,:);
    Y_temp{i} = Y(s:e,:);
    
    % prep next
    s = e+1;
    e = s+batch_size(i)-1;
    if i == num_batches-1 && e ~= n
        e = n;
        batch_size(num_batches) = e-s+1;
    end
end

parfor i = 1:num_batches
    [ LL_temp(i), ~, Y_prob_temp{i} ] = compute_loglikelihood_sequential( Experts, gate, X_temp{i}, Y_temp{i}, is_switching );
end

Y_prob = [];
for i = 1:num_batches
    Y_prob = [Y_prob;Y_prob_temp{i}];
end

LL = sum(LL_temp);
avg_prob = mean(Y_prob);
%avg_prob = batch_size*avg_prob_temp'/n;

end%end-of-function compute_loglikelihood_MT_parallel()

