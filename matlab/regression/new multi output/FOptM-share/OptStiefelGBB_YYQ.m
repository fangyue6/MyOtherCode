function [A] = OptStiefelGBB_YYQ(A, B, X, Y, b, e, LH, alpha)

function [F, G] = fun(A, B, X, Y, b, e, LH, alpha)
  G = 2*(-Y'*X*B + A*B'*X'*X*B + b'*e'*X*B + alpha*A*B'*X'*LH*X*B);%µ¹Êý
  F = norm(Y - X*B*A' - e*b, 'fro')^2 + alpha * trace(A*B'*X'*LH*X*B*A');%Ô­º¯Êý
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


tic; [A, out]= OptStiefelGBB(A, @fun, opts, B, X, Y, b, e, LH, alpha); tsolve = toc;
out.fval = -2*out.fval; % convert the function value to the sum of eigenvalues
% fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, cpu: %f, norm(BT*B-I): %3.2e \n', ...
%             out.fval, out.itr, out.nfe, tsolve, norm(B'*B - eye(k), 'fro') );

end
