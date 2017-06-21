% Mixture of CTBN, EM algorithm
function [Experts, gate, gate_lambda, Q_over_time, LL_train_over_time, LL_test_over_time, t_testing] = MCTBN_learn_ME(X, Y, Experts, is_switching, X_test, Y_test, max_iter, gate_lambda )

% Configuration
is_verbose = true;
%is_learn_lambda = true;
is_init_train = false;  % init_train: true, if initial model train is required (won't be necessary if boosting is ready)

if ~exist('is_switching','var')
    is_switching = true;
end

if exist('X_test','var') && ~isempty(X_test)
    is_testing = true;
else
    is_testing = false;
end

optimization_method = 'LBFGS';
%optimization_method = 'GD';
%optimization_method = 'L1OPT';


% PREP: variable init
%lambda_over_time = [];
Q_over_time = [];
LL_train_over_time = [];
LL_test_over_time = [];

t_testing = 0;

K = length(Experts);
[N, m] = size(X);

TOL = 1e-3;

if ~exist('max_iter','var')
    MAXITER = 250;
else
    MAXITER = max_iter;
end

% init
%if exist('init_gate','var')
%    disp('supplied');
%    gate = init_gate;
%else
%    fprintf(2, 'warn: no initial gate parameters are provided; by default,\n\tgate = [zeros(K,m) ones(K,1)/K]' );

fprintf( 'msg (MLME-CTBN): using gate = uniform; gate = [zeros(K,m) ones(K,1)/K]\n' );
gate = [zeros(K,m) ones(K,1)/K];
    %gate = ones(K,m+1)/(m+1);
%end

n_iter = 0;
Q = 0;

for k = 1:K
    if is_init_train    % in case the CTBN parameters are not set
        if is_switching
            Experts{k} = compute_tree_weights(Experts{k}, X, Y, '-s 1 -c 1');
        else
            Experts{k} = compute_tree_weights(Experts{k}, X, Y, '-s 0 -c 1');
        end
    end
end

% % init choose lambda
% if is_learn_lambda
%     fprintf( 'msg: choosing gate_lambda\n' );
%     gate_lambda = choose_lambda( Experts, gate, X, Y, is_switching, optimization_method );
% end

% ECLL
h_x = MCTBN_compute_gate_posterior_h_x(Experts, gate, X, Y, is_switching);
Q = MCTBN_compute_ECLL_ME( Experts, gate, h_x, X, Y, is_switching );


% bookkeeping
%lambda_over_time(:,1) = lambda;
Q_over_time(1) = Q;
if is_testing
    t_test_start = clock;
    LL_train_over_time(1) = compute_loglikelihood_ME(Experts, gate, X, Y, is_switching);
    LL_test_over_time(1) = compute_loglikelihood_ME(Experts, gate, X_test, Y_test, is_switching);
    %fprintf( '%f / %f / %f\n', Q, LL_train_over_time(1), LL_test_over_time(1) );
    t_test_end = clock;
    t_testing = t_testing + etime(t_test_end,t_test_start);
end

% if is_verbose
%     fprintf( 'n_CTBNs = %d\n', K );
%     fprintf( 'init: %.2f\t', Q );
%     fprintf( ' | LL_tr: %.2f\t  | LL_ts: %.2f\n', LL_train_over_time(1), LL_test_over_time(1) );
% end


% EM
is_converged = false;

while ~is_converged && n_iter < MAXITER
    n_iter = n_iter + 1;
    
    % Expectation - update h_k(i)
    h_x = MCTBN_compute_gate_posterior_h_x(Experts, gate, X, Y, is_switching);
    
    % ECLL
    q_tmp1 = MCTBN_compute_ECLL_ME( Experts, gate, h_x, X, Y, is_switching );

    lin_gate = reshape(gate, [], 1);
    ECLL = compute_F1(lin_gate, h_x, X);
    
    % Maximization
    % f1: Gating network learning - L-BFGS
    if strcmp(optimization_method, 'LBFGS')
        gate = learn_gate_parameters_LBFGS( gate, h_x, X, gate_lambda );
    elseif strcmp(optimization_method, 'GD')
        is_accelerated = false;
        gate = learn_gate_parameters_GD( gate, h_x, X, 5, is_accelerated );
    elseif strcmp(optimization_method, 'L1OPT')
        gate = learn_gate_parameters_L1OPT( gate, h_x, X );
    else
        fprintf( 2, 'err (MLME-CTBN): unknown optimization method is specified.\n' );
    end
    
    % f2: CTBN paramester learning - weighted LR train
    for k = 1:K
        w = h_x(:,k)/sum(h_x(:,k));
        if is_switching
            %Experts{k} = compute_tree_weights(Experts{k}, X, Y, h_x(:,k), '-s 1 -l 0');
            Experts{k} = compute_tree_weights(Experts{k}, X, Y, w, '-s 1 -l 0');
        else
            %Experts{k} = compute_tree_weights(Experts{k}, X, Y, h_x(:,k), '-s 0 -l 0');
            Experts{k} = compute_tree_weights(Experts{k}, X, Y, w, '-s 0 -l 0');
        end
    end
    
    
    % score Q
    % ECLL
    Q_new = MCTBN_compute_ECLL_ME( Experts, gate, h_x, X, Y, is_switching );
    
        
    
    if is_testing
        t_test_start = clock;
        LL_train_over_time(n_iter+1) = compute_loglikelihood_ME(Experts, gate, X, Y, is_switching);
        LL_test_over_time(n_iter+1) = compute_loglikelihood_ME(Experts, gate, X_test, Y_test, is_switching);
        %fprintf( '%f / %f / %f\n', Q_new, LL_train_over_time(n_iter+1), LL_test_over_time(n_iter+1) );
        t_test_end = clock;
        t_testing = t_testing + etime(t_test_end,t_test_start);
    end

    % continue?
    is_converged = Q_new - Q < TOL;
    
    % screen dump
    if is_verbose
        fprintf( '%d(%d): %.2f -> %.2f\n', n_iter, is_converged, Q, Q_new );
        %fprintf( ' | LL_tr: %.2f\t  | LL_ts: %.2f\n ', LL_train_over_time(n_iter+1), LL_test_over_time(n_iter+1) );
    end
    
    % bookkeeping & update Q
    Q_over_time(n_iter+1) = Q_new;
    Q = Q_new;
end


if is_converged
    fprintf('msg (MLME-CTBN): Converged in %d steps.\n', n_iter);
else
    fprintf('msg (MLME-CTBN): Not converged in %d steps.\n', MAXITER);
end


if is_testing
    fprintf(2,'msg (MLME-CTBN): spent %.2f sec for testing.\n', t_testing );
end

end%end-of-function: learn_output_tree_ME()

