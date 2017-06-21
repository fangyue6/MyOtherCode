function [Y_pred, Y_log_prob, gate_xi] = MAP_prediction_MCC_SA( CC, gate, X, Y, max_iter )

if ~exist('max_iter','var')
    max_iter = 150;
end

is_verbose = false;
K = length(CC);
[N,d] = size(Y);
Y_pred = zeros(N, d);
Y_log_prob = zeros(N, 1);
gate_xi = zeros(N, K);

for i = 1:N
    x = X(i,:);
    y = Y(i,:);
    exp_theta_x = zeros(1,K);
    
    % calc gate output
    for k = 1:K
        exp_theta_x(k) = exp(dot( gate(k,:), [x 1] ));
    end
    g_xi = exp_theta_x / sum(exp_theta_x);
    
    % s=pred, s_new=pred_new; e=prob, e_new=prob_new
    % opt1: choose best lambda
%     [~,k0] = max(lambda);
%     s0 = MAP_prediction( Trees{k0}, x, is_switching );
    
    % opt2: choose best prob
    s0_pred = cell(1,K);
    s0_prob = zeros(1,K);
    for k=1:K
        s0_pred{k} = CC_predict(CC{k}.model, x, y, CC{k}.permutation);
        s0_prob(k) = compute_energy( CC, g_xi, x, s0_pred{k} );
    end
    [~,s0_best] = min(s0_prob);
    s0 = s0_pred{s0_best};

    % opt3: dummy (all-0)
    %s0 = zeros( 1, length(Trees{k}) );
    
    ObjFn = @(s) compute_energy( CC, g_xi, x, s );
        options = saoptimset('DataType', 'custom');
        options = saoptimset(options, 'TemperatureFcn', @temperatureexp, 'InitialTemperature', 100);
        options = saoptimset(options, 'AnnealingFcn', @neighbor);
        options = saoptimset(options, 'AcceptanceFcn', @acceptancesa);
        
        options = saoptimset(options, 'MaxIter', max_iter);
        options = saoptimset(options, 'ReannealInterval', 50);
        
        if is_verbose
            options = saoptimset(options, 'Display', 'iter', 'DisplayInterval', 100);
        else
            options = saoptimset(options, 'Display', 'off');
        end
        
    [s_best, fval, exitFlag, output] = simulannealbnd(ObjFn, s0, [], [], options);
    
    Y_pred(i,:) = s_best;
    Y_log_prob(i,1) = log(-compute_energy( CC, g_xi, x, y ));
    gate_xi(i,:) = g_xi;
end

end %end-of-function MAP_prediction_MT_SA()


function e = compute_energy( CC, g_x, x, s )
% energy = - gx .* prob
K = length(CC);
tmp_prob = zeros(1,K);
for k = 1:K
    tmp_prob(k) = exp(evaluate_PCC_log_likelihood(CC{k}.model, x, s, CC{k}.permutation));
end
e = -dot(tmp_prob, g_x);

end %end-of-function compute_prob_MT()


function s_new = neighbor(optimvalues,problem)

s = optimvalues.x;
d = length(s);

s_new = s;
i = randi(d,1,1);
s_new(i) = -(s_new(i)-1);   %flip
end %end-of-function neighbor()



% function s_new = neighbor_n(optimvalues,problem)
% 
% s = optimvalues.x;
% d = length(s);
% 
% n_flips = round(d/2 * mean(optimvalues.temperature)/100);
% if n_flips == 0, n_flips = 1; end
% 
% s_new = s;
% j = 0;
% is_flipped = zeros(1,d);
% while j < n_flips
%     i = randi(d,1,1);
%     if ~is_flipped(i)
%         s_new(i) = -(s_new(i)-1);   %flip
%         is_flipped(i) = 1;
%         j = j + 1;
%     end
% end
% end %end-of-function neighbor()


