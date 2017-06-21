function [ h_x ] = MCC_compute_gate_posterior_h_x(Experts, gate, X, Y)

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
    h_x = MCC_compute_gate_h_x_parallel( Experts, gate, X, Y);
else
    h_x = MCC_compute_gate_h_x_sequential( Experts, gate, X, Y);
end

end%end-of-function: compute_h_x()



function [ h_x ] = MCC_compute_gate_h_x_sequential(Experts, gate, X, Y)

N = size(X,1);
K = length(Experts);

h_x = zeros(N,K);

% update h_k(i)
for i = 1:N
    tmp_prob = zeros(1,K);
    exp_theta_x = zeros(1,K);

    for k = 1:K
        tmp_prob(k) = exp(evaluate_PCC_log_likelihood(Experts{k}.model, X(i,:), Y(i,:), Experts{k}.permutation));
        exp_theta_x(k) = exp(dot( gate(k,:), [X(i,:) 1] ));
    end

    g_xi = exp_theta_x / sum(exp_theta_x);

    h_xi = g_xi .* tmp_prob;
    h_x(i,:) = h_xi / sum(h_xi);
end

end%end-of-function: compute_h_x_sequential()



function [ h_x ] = MCC_compute_gate_h_x_parallel(Experts, gate, X, Y)

% PREP: parallelization (divide batch sets)
global num_batches;

X_temp = cell(1,num_batches);
Y_temp = cell(1,num_batches);

batch_size = round(N/num_batches);
batch_size = repmat(batch_size, [1 num_batches]);

s = 1;
e = s+batch_size(1)-1;
for i = 1:num_batches
%    par_idx{i} = (s:e)';
    X_temp{i} = X(s:e,:);
    Y_temp{i} = Y(s:e,:);

    % prep next
    s = e+1;
    e = s+batch_size(i)-1;
    if i == num_batches-1 && e ~= N
        e = N;
        batch_size(num_batches) = e-s+1;
    end
end

% update h_k(i) on each chop
tmp_h_x = cell(num_batches,1);
parfor j = 1:num_batches
    tmp_h_x{j} = MCC_compute_gate_h_x_sequential(Experts, gate, X, Y);
end

h_x = [];
for j = 1:num_batches
    h_x = [ h_x; tmp_h_x{j} ];
end

end