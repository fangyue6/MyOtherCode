% min_{W'*W=I,b}  \sum_i ||(I-W*W')*(xi-b)||_2
% solved with reweighted method
% written by Feiping Nie
function [W, b, obj] = RPCA_OM(X, feature_num, NITER)
% X: dim*n data matrix, each column is a data point
% feature_num: reduced dimension
% NITER: iteration number
% W: learned projection matrix
% b: learned optimal mean
% obj: objective values in the iterations

% For more details, please see:
% Feiping Nie, Jianjun Yuan, Heng Huang.
% Optimal Mean Robust Principal Component Analysis.
% The 31st International Conference on Machine Learning (ICML), 2014.


[~, n] = size(X);

if nargin <= 2
    NITER = 10;
end;

d = ones(n,1);
obj = zeros(NITER,1);
for iter = 1:NITER
    D = spdiags(sqrt(d),0,n,n);
    b = X*d/sum(d);
    A = X - b*ones(1,n);
    M = A*D;
    [W, ~] = svds(M,feature_num);
    B = A - W*(W'*A);
    Bi = sqrt(sum(B.*B,1)+eps)';
    d = 0.5./(Bi);   
    obj(iter) = sum(Bi);
end;
