% solving min_{X}  ||A*X-Y||^2 + r1 * ||X||_1 + r2 * ||X||_21
% written by Feiping Nie
function [X, obj]=L2R1R21(A, Y, r1, r2, X0)
% A: n*m matrix
% Y: n*c matrix
% r1,r2: regularization parameters
% X0: initial X (optional)
% X: m*c matrix
% obj: objective values

% Ref:
% Hua Wang, Feiping Nie, Heng Huang, Shannon Risacher, Chris Ding, Andrew J. Saykin, Li Shen, ADNI.
% Sparse Multi-Task Regression and Feature Selection to Identify Brain Imaging Predictors for Memory Performance.
% The 13th International Conference on Computer Vision (ICCV), Barcelona, Spain, 2011.



ITER = 50;
[n, m] = size(A);
c = size(Y,2);

if nargin<5
    B = ones(m,c);
    D2 = diag(ones(m,1));
else
    X = X0;
    Xi1 = sqrt(X.*X+eps);
    B = 0.5./Xi1;
    Xi2 = sqrt(sum(X.*X,2)+eps);
    d2 = 0.5./(Xi2);
    D2 = diag(d2);
end;

AA = A'*A;
AY = A'*Y;
obj = zeros(ITER,1);
for iter = 1:ITER
    for i=1:c
        D1 = diag(B(:,i));
        M = AA+r1*D1+r2*D2;
        X(:,i) = M\AY(:,i);
    end;
    
    Xi1 = sqrt(X.*X+eps);
    B = 0.5./Xi1;
    Xi2 = sqrt(sum(X.*X,2)+eps);
    d2 = 0.5./(Xi2);
    D2 = diag(d2);
    
    X1 = sum(Xi1(:));
    X21 = sum(Xi2);
    
    obj(iter) = trace((A*X-Y)'*(A*X-Y)) + r1*X1 + r2*X21;
end;





