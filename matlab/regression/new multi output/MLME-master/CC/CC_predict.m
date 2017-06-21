%% CC_test: the classifier chain model [Read et al, 2009]
function [Y_pred, Y_log_prob, Yi_log_prob] = CC_predict(M, X, Y, P)

d = size(M,2);
n = size(X,1);

prob = zeros(n, d);
for i = 1:d
    %prob(:,i) = LR_predict(M{i}, [X_ts round(prob(:,1:i-1))]);
    % hardwiring
    prob(:,P(i)) = LR_predict(M{i}, [X round(prob(:,P(1:i-1)))]);
    
end

%fix for multi-class
Y_pred = round(prob);

Y_log_prob = zeros(n,1);
Yi_log_prob = zeros(n,d);
for i=1:n
     [Y_log_prob(i,1), ~, Yi_log_prob(i,:)] = compute_log_prob(Y(i,:), prob(i,:));
end

% 
% for q=1:n
%     Y_log_prob(q) = evaluate_log_likelihood(M, X(q,:), Y(q,:), P);
% end
% 
% end %end of function CC_predict()
% 
% 
% function [ l ] = evaluate_log_likelihood(M, x, y, P)
%     d = size(M,2);
%     for i=1:d
%         temp_prob(P(i)) = LR_predict(M{i}, [x y(P(1:i-1))]);
%     end
%     l = compute_log_prob(y, temp_prob);
% end