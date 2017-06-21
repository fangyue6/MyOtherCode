% Mixture of CTBN, EM algorithm
%function [Experts, gate, gate_lambda, Q_over_time, LL_train_over_time, LL_test_over_time, t_testing] = learn_output_tree_ME(X, Y, Experts, X_test, Y_test, max_iter, gate_lambda )

function [Experts, gate, t_testing] = MCC_learn_ME( Experts, X, Y, X_test, Y_test, max_iter, gate_lambda )

% Configuration
is_verbose = true;

%is_learn_lambda = true;
is_init_train = false;  % init_train: true, if initial model train is required (won't be necessary if boosting is ready)

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
fprintf( 'msg (MLME-CC): using gate = uniform; gate = [zeros(K,m) ones(K,1)/K]\n' );
gate = [zeros(K,m) ones(K,1)/K];

n_iter = 0;
Q = 0;

for k = 1:K
    if is_init_train    % in case the CTBN parameters are not set
        Experts{k} = CCs2_train(X, Y);
    end
end

% % init choose lambda
% if is_learn_lambda
%     fprintf( 'msg: choosing gate_lambda\n' );
%     gate_lambda = choose_lambda( Experts, gate, X, Y, optimization_method );
% end


% ECLL
h_x = MCC_compute_gate_posterior_h_x(Experts, gate, X, Y);
Q = MCC_compute_ECLL_ME( Experts, gate, h_x, X, Y );

%lambda_over_time(:,1) = lambda;    % bookkeeping
Q_over_time(1) = Q; % bookkeeping

if is_testing
    t_test_start = clock;
    LL_train_over_time(1) = compute_loglikelihood_ME(Experts, gate, X, Y);
    LL_test_over_time(1) = compute_loglikelihood_ME(Experts, gate, X_test, Y_test);
    %fprintf( '%f / %f / %f\n', Q, LL_train_over_time(1), LL_test_over_time(1) );
    t_test_end = clock;
    t_testing = t_testing + etime(t_test_end,t_test_start);
end

% if is_verbose
%     fprintf( 'n_Experts = %d\n', K );
%     fprintf( 'init: %.2f\t', Q );
%     fprintf( ' | LL_tr: %.2f\t  | LL_ts: %.2f\n', LL_train_over_time(1), LL_test_over_time(1) );
% end



% EM
is_converged = false;
while ~is_converged && n_iter < MAXITER
    n_iter = n_iter + 1;
    
    last_Experts = Experts;
    last_gate = gate;
    
    % Expectation - update h_k(i)
    h_x = MCC_compute_gate_posterior_h_x(Experts, gate, X, Y);
        
%         % ECLL
%         q_tmp1 = compute_ECLL_ME( Experts, gate, h_x, X, Y );
%         
%         lin_gate = reshape(gate, [], 1);
%         ECLL = compute_F1(lin_gate, h_x, X);
        
        
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
        fprintf( 2, 'err (MLME-CC): unknown optimization method is specified.\n' );
    end
%         if is_verbose
%             fprintf( '\n(2) AFTER GATE UP   | ' );
%             %fprintf( '\t|  (2) %.2f(-%f=%f)', q_tmp2, ECLL, q_tmp2-ECLL );
%             if q_tmp1 > q_tmp2
%                 fprintf(2, '\b* \n');
%             end
%         end
%         % ECLL
%         q_tmp2 = compute_ECLL_ME( Experts, gate, h_x, X, Y );
%         
%         lin_gate = reshape(gate, [], 1);
%         ECLL = compute_F1(lin_gate, h_x, X);
        
        
    
    % f2: CTBN paramester learning - weighted LR train
    for k = 1:K
        w = h_x(:,k)/sum(h_x(:,k)) * N;
        [Experts{k}.model, Experts{k}.permutation] = CC_train_weighted(X, Y, w, Experts{k}.permutation);
    end
    
    % score Q
    % ECLL
    Q_new = MCC_compute_ECLL_ME( Experts, gate, h_x, X, Y );
    
        
    
    if is_testing
        t_test_start = clock;
        LL_train_over_time(n_iter+1) = compute_loglikelihood_ME(Experts, gate, X, Y);
        LL_test_over_time(n_iter+1) = compute_loglikelihood_ME(Experts, gate, X_test, Y_test);
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


% Experts = last_Experts;
% gate = last_gate;

if is_converged
    fprintf('msg (MLME-CC): Converged in %d steps.\n', n_iter);
else
    fprintf('msg (MLME-CC): Not converged in %d steps.\n', MAXITER);
end


if is_testing
    fprintf(2,'msg (MLME-CC): spent %.2f sec for testing.\n', t_testing );
end

end%end-of-function: learn_output_tree_ME()

