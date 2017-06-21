% isbias=1: min_{Z,b}  ||X-b*1'-Z||_21 + r*||Z||_*
% isbias=0: min_{Z}  ||X-Z||_21 + r*||Z||_*
% solved with ALM method in this code, but can also be solved with reweighted method
% written by Feiping Nie
function [Z,b,obj,eigv,Lambda_max] = CRPCA_OM(X, r, isbias, mu, rho, NITER)
% X: dim*n data matrix, each column is a data point
% r: regularization parameter
% isbias: 1 for optimal mean convex RPCA, 0 for previous convex RPCA
% mu, rho: ALM parameters, mu>0, rho>1, larger mu and rho, faster algorithm, but less accurate
% NITER: iteration number, it is often required that mu*(rho^NITER)>1000
% Z: learned data matrix
% b: learned optimal mean if isbias=1, and 0 if isbias=0
% obj: objective values in the iterations, it is not guaranteed to be monotonically decreased in ALM
% eigv: singular values of the learned data matrix Z
% Lambda_max: the ALM solver is converged if Lambda_max is very small

% For more details, please see:
% Feiping Nie, Jianjun Yuan, Heng Huang.
% Optimal Mean Robust Principal Component Analysis.
% The 31st International Conference on Machine Learning (ICML), 2014.


[dim, n] = size(X);
H = eye(n) - 1/n*ones(n);

if nargin <= 3
    mu = 0.1;
    rho = 1.1;
    NITER = 100;
end;

Lambda = zeros(dim, n);
E = rand(dim, n);

obj = zeros(NITER,1);
for iter = 1:NITER
    
    inmu = 1/mu;
    X1 = X - E + inmu*Lambda;
    r1 = r*inmu;
    if isbias == 1
        b = mean(X1,2);
        XH = X1*H;
        [U, A, V] = svd(XH,'econ');
    else
        b = zeros(dim,1);
        [U, A, V] = svd(X1,'econ');
    end;
    a = diag(A);
    a1 = max(a-r1, 0);
    Z = U*diag(a1)*V';
    
    X2 = X - b*ones(1,n) - Z + inmu*Lambda;
    r2 = inmu;
    for i=1:n
        x = X2(:,i);
        aa = norm(x);
        if aa > r2
            E(:,i) = (1-r2/aa)*x;
        else
            E(:,i) = 0;
        end;
    end;
    Lambda = Lambda + mu*(X - b*ones(1,n) - Z - E);
    mu = min(10^8,rho*mu);

    % compute objective value
    if isbias == 1
        XHZ = XH-Z;
    else
        XHZ = X-Z;
    end;
    err = sum(sqrt(sum(XHZ.^2)));
    obj(iter) = err + r*sum(a1);
end;
eigv = a1;

Lambda_max = max(abs(Lambda(:)))/mu;

