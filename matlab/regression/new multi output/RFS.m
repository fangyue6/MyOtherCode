function [W, obj]=RFS(X, Y, r, X0)
% function [W, obj]=L21R21_inv(X, Y, r, X0)
%% 21-norm loss with 21-norm regularization

%% Problem
%
%  min_X  || A X - Y||_21 + r * ||X||_21

% Ref: Feiping Nie, Heng Huang, Xiao Cai, Chris Ding. 
% Efficient and Robust Feature Selection via Joint L21-Norms Minimization.  
% Advances in Neural Information Processing Systems 23 (NIPS), 2010.


% Y = A;
NIter = 50;
[m n] = size(X);
if nargin < 4
    d = ones(n,1);
    d1 = ones(m,1);
else
    Xi = sqrt(sum(X0.*X0,2));
    d = 2*Xi;
    AX = X*X0-Y;
    Xi1 = sqrt(sum(AX.*AX,2)+eps);
    d1 = 0.5./Xi1;
end;

if m>n
    for iter = 1:NIter
        D = spdiags(d,0,n,n);
        D1 = spdiags(d1,0,m,m);
        DAD = D*X'*D1;
        W = (DAD*X+r*eye(n))\(DAD*Y);

        Xi = sqrt(sum(W.*W,2));
        d = 2*Xi;

        AX = X*W-Y;
        Xi1 = sqrt(sum(AX.*AX,2)+eps);
        d1 = 0.5./Xi1;

        obj(iter) = sum(Xi1) + r*sum(Xi);
    end;
else
    for iter = 1:NIter
        D = spdiags(d,0,n,n);
        D1 = spdiags(d1,0,m,m);
        DAD = D*X'*D1;
        W = DAD*((X*DAD+r*eye(m))\Y);

        Xi = sqrt(sum(W.*W,2));
        d = 2*Xi;

        AX = X*W-Y;
        Xi1 = sqrt(sum(AX.*AX,2)+eps);
        d1 = 0.5./Xi1;

        obj(iter) = sum(Xi1) + r*sum(Xi);
    end;
end;
1;