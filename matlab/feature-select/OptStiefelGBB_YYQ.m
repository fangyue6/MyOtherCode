function [W] = OptStiefelGBB_YYQ( W, X, Y, L, P, S, lambda_1, lambda_2, lambda_3)

function [F, G] = fun( W, X, Y, L, P, S, lambda_1, lambda_2, lambda_3)
  G = 2*(X'*L*X*W+lambda_1*(X'*X*W-X'*Y)+lambda_3*(P*W)); %导数
  F = trace(W'*X'*L*X*W) + lambda_1*norm(Y-X*W, 'fro')^2 + lambda_2*sum(sum(S.^2)) + lambda_3*trace(W'*P*W);  %原函数
end

% n = 1000; k = 6;
% M = randn(n); 
% M = M'*M;
% N = randn(k,n);
% B0 = randn(n,k);    
% B0 = orth(B0);

opts.record = 0; %
opts.mBitr  = 1000;
opts.Btol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;


tic; [W, out]= OptStiefelGBB(W, @fun, opts, X, Y, L, P, S, lambda_1, lambda_2, lambda_3); tsolve = toc;
out.fval = -2*out.fval; % convert the function value to the sum of eigenvalues
% fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, cpu: %f, norm(BT*B-I): %3.2e \n', ...
%             out.fval, out.itr, out.nfe, tsolve, norm(B'*B - eye(k), 'fro') );

end
