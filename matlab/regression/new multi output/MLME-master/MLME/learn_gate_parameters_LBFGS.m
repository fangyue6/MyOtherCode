function [ gate ]= learn_gate_parameters_LBFGS( gate0, h_x, X, gate_lambda, max_iter )

[~, m] = size(X);

% if lambda is not set, set it to 1
if ~exist( 'gate_lambda', 'var' )
    gate_lambda = 1;
end

% linearize gate0
lin_gate0 = reshape(gate0', [], 1);

% L-BFGS
options = [];
options.display = 'none';
if exist( 'max_iter', 'var' )
    options.maxFunEvals = max_iter;
else
    options.maxFunEvals = 100;
end
%options.maxIter = 1;
options.Method = 'lbfgs';

ObjFn = @(lin_gate)eval(lin_gate, h_x, X, gate_lambda);

% % 3-folds cross validation
% FOLD = 3;
% indices = crossvalind('Kfold',X(:,1),FOLD);
% for fold = 1:FOLD
%     X_tr = X(indices ~= fold);
%     X_val = X(indices == fold);
% end

lin_gate = minFunc(ObjFn, lin_gate0, options);

% lin -> mat: gate
gate = reshape(lin_gate, m+1, [])';

end


function [ F, dF ] = eval(lin_gate, h_x, X, gate_lambda)

% compute ECLL & gradient
[ECLL, lin_grad] = compute_F1(lin_gate, h_x, X, gate_lambda);
F = -ECLL;
dF = -lin_grad;

end



% function f1 = compute_f1( h_x, g_x )
% 
% N = size(h_x, 1);
% f1 = 0;
% 
% for i = 1:N
%     f1 = f1 + sum( h_x(i,:) .* log(g_x(i,:)) );
% end
% end




% function [g_x] = compute_g_x(gate, X)
% 
% N = size( X, 1 );
% K = size( gate, 1 );
% 
% % global runParallel;
% % global num_batches;
% 
% % compute g_x
% % if runParallel && num_batches <= size(X,1)
% %     tmp_g_x = cell(num_batches,1);
% %     parfor j = 1:num_batches
% %         tmp_g_x{j} = zeros(batch_size(j),K)
% % 
% %         for i = 1:batch_size(j)
% %             exp_theta_x = zeros(1,K);
% % 
% %             for k = 1:K
% %                 exp_theta_x(k) = exp(dot( gate(k,:), [X(i,:) 1] ));
% %             end
% %             tmp_g_x{j}(i,:) = exp_theta_x / sum(exp_theta_x);
% %         end
% %     end
% % 
% %     g_x = [];
% %     for j = 1:num_batches
% %         g_x = [ g_x; tmp_g_x{j} ];
% %     end
% %     
% % else
%     g_x = zeros(N,K);
% 
%     for i = 1:N
%         exp_theta_x = zeros(1,K);
% 
%         for k = 1:K
%             exp_theta_x(k) = exp(dot( gate(k,:), [X(i,:) 1] ));
%         end
% 
%         g_x(i,:) = exp_theta_x / sum(exp_theta_x);
%     end
% % end
% 
% end
