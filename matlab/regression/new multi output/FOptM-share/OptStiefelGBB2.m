function [A] = OptStiefelGBB2(A0,O,P)

function [F, G] = fun(A,  O,  P)
  G = O*A +O'*A - 2*P;
  F = sum(dot(G,A,1));
end

% n = 1000; k = 6;
% O = randn(n); 
% O = O'*O;
% N = randn(k,n);
% A0 = randn(n,k);    
% A0 = orth(A0);

opts.record = 0; %
opts.OAitr  = 1000;
opts.Atol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;


tic; [A, out]= OptStiefelGBB(A0, @fun, opts, O,P); tsolve = toc;
out.fval = -2*out.fval; % convert the function value to the suO of eigenvalues
% fprintf('\nOptO: oAj: %7.6e, itr: %d, nfe: %d, cpu: %f, norO(AT*A-I): %3.2e \n', ...
%             out.fval, out.itr, out.nfe, tsolve, norO(A'*A - eye(k), 'fro') );

end
