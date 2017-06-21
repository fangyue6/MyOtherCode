function [g_x] = compute_gate_g_x(gate, X)

N = size( X, 1 );
K = size( gate, 1 );

% global runParallel;
% global num_batches;

% compute g_x
% if runParallel && num_batches <= size(X,1)
%     tmp_g_x = cell(num_batches,1);
%     parfor j = 1:num_batches
%         tmp_g_x{j} = zeros(batch_size(j),K)
% 
%         for i = 1:batch_size(j)
%             exp_theta_x = zeros(1,K);
% 
%             for k = 1:K
%                 exp_theta_x(k) = exp(dot( gate(k,:), [X(i,:) 1] ));
%             end
%             tmp_g_x{j}(i,:) = exp_theta_x / sum(exp_theta_x);
%         end
%     end
% 
%     g_x = [];
%     for j = 1:num_batches
%         g_x = [ g_x; tmp_g_x{j} ];
%     end
%     
% else
    g_x = zeros(N,K);

    for i = 1:N
        exp_theta_x = zeros(1,K);

        for k = 1:K
            exp_theta_x(k) = exp(dot( gate(k,:), [X(i,:) 1] ));
        end

        g_x(i,:) = exp_theta_x / sum(exp_theta_x);
    end
% end

end