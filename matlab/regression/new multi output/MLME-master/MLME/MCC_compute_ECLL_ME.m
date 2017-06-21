%LL: ECLL (Q in EM) of [X,Y]
%avg_prob: the average probability for the true labels P(yi|xi)
function [ Q ] = MCC_compute_ECLL_ME( Experts, gate, h_x, X, Y )

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
    [ Q ] = compute_ECLL_ME_parallel( Experts, gate, h_x, X, Y );
else
    [ Q ] = compute_ECLL_ME_sequential( Experts, gate, h_x, X, Y );
end

end%end-of-function compute_ECLL_ME()


%%%%
function [ Q ] = compute_ECLL_ME_sequential( Experts, gate, h_x, X, Y )

K = length(Experts);
N = size(X,1);

g_x = compute_gate_g_x(gate,X);

Q_f1 = 0;
Q_f2 = 0;
for k = 1:K
    for i = 1:N
        tmp_prob = exp(evaluate_PCC_log_likelihood(Experts{k}.model, X(i,:), Y(i,:), Experts{k}.permutation));
        
        Q_f1 = Q_f1 + h_x(i,k) * log(g_x(i,k));
        Q_f2 = Q_f2 + h_x(i,k) * log(tmp_prob);
    end
end

Q = Q_f1 + Q_f2;

end%end-of-function compute_ECLL_ME()


%%%%
function [ Q ] = compute_ECLL_ME_parallel( Experts, gate, h_x, X, Y )

global num_batches;

N = size(X,1);

batch_size = round(N/num_batches);
batch_size = repmat(batch_size, [1 num_batches]);

s = 1;
e = s+batch_size(1)-1;
for i = 1:num_batches
    X_temp{i} = X(s:e,:);
    Y_temp{i} = Y(s:e,:);
    h_x_temp{i} = h_x(s:e,:);
    
    % prep next
    s = e+1;
    e = s+batch_size(i)-1;
    if i == num_batches-1 && e ~= N
        e = N;
        batch_size(num_batches) = e-s+1;
    end
end

parfor i = 1:num_batches
    [ tmp_Q(i) ] = compute_ECLL_ME_sequential( Experts, gate, h_x_temp{i}, X_temp{i}, Y_temp{i} );
end

Q = sum(tmp_Q);

end%end-of-function compute_ECLL_ME()

