function [B] = OptStiefelGBB1(B0,M,N)

function [F, G] = fun(B,  M,  N)
  G = M*B +M'*B - 2*N;
  F = sum(dot(G,B,1));
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


tic; [B, out]= OptStiefelGBB(B0, @fun, opts, M,N); tsolve = toc;
out.fval = -2*out.fval; % convert the function value to the sum of eigenvalues
% fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, cpu: %f, norm(BT*B-I): %3.2e \n', ...
%             out.fval, out.itr, out.nfe, tsolve, norm(B'*B - eye(k), 'fro') );

end
