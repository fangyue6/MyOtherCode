function [ F, dF_linearized ] = compute_F1( lin_gate, h_x, X, lambda )

[N, m] = size(X);
K = size(h_x, 2);

% g_k(i)
gate = reshape(lin_gate, m+1, [])';
g_x = compute_gate_g_x(gate, X);

% F = ECLL
F = 0;
for i = 1:N
    %F = F + sum( h_x(i,:) .* log(g_x(i,:)) );
    F = F + dot(h_x(i,:), log(g_x(i,:)));
end

% dF = gradient
diff = h_x - g_x;

% dF
dF = diff'*[X ones(N,1)];

% regularizer
if exist('lambda','var')
    % regularize over the gate params for X only
    gate_param_on_x = [ gate(:,1:m) zeros(K,1) ];
    lin_gate_param_on_x = reshape( gate_param_on_x', [], 1 );
    
    F = F - lambda/2 * sum(lin_gate_param_on_x .^ 2);
    dF = dF - lambda*gate_param_on_x;
    
    % regularize over the gate params for X and bias
    %F = F - lambda/2 * sum(lin_gate .^ 2);
    %dF = dF - lambda*gate;
end
dF_linearized = reshape( dF', [], 1 );

    % dF_linearized = [];
    % for j = 1:K
    %     dFj = zeros(1, m+1);
    %     for i = 1:N
    %         dFj = dFj + diff(i,j) * [X(i,:) 1];
    %     end
    %     dF_linearized = [dF_linearized; -dFj'];
    % end

end