function test_gang

function [F, G] = fun(X,  A)
  G = -(A*X);%µ¹Êý
  F = 0.5*sum(dot(G,X,1));%Ô­º¯Êý
end

n = 1000; k = 6;
A = randn(n); A = A'*A;
opts.record = 0; %
opts.mxitr  = 1000;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;

X0 = randn(n,k);    X0 = orth(X0);
tic; [X, out]= OptStiefelGBB(X0, @fun, opts, A); tsolve = toc;
out.fval = -2*out.fval; % convert the function value to the sum of eigenvalues
fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, cpu: %f, norm(XT*X-I): %3.2e \n', ...
            out.fval, out.itr, out.nfe, tsolve, norm(X'*X - eye(k), 'fro') );

end
