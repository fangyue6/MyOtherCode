function [rank, score] = TRCFS_semi(X, Y, L, alpha)
% X: n*dim data matrix, each row is a data point
% Y: n*c label matrix, if the i-th data point is labeled as j, then Y(i,j)=1, otherwise Y(i,j)=0
% L: n*n Laplacian matrix, which can be achieved by Gassuial function, LLE, LTSA, LSE, LRGA or other manifold learning approaches
% alpha: parameter, 0<alpha<1
% rank: the index of ranked features 
% score: ranking scores

% Ref:
% Yun Liu, Feiping Nie$^*$, Jigang Wu, Lihui Chen.
% Efficient Semi-supervised Feature Selection with Noise Insensitive Trace Ratio Criterion.
% Neurocomputing, 105:12-18, 2013. 


if nargin < 4
    alpha = 0.99;
end;

n = size(X,1);
X0 = X;
mX0 = mean(X0);
X1 = X0 - ones(n,1)*mX0;
scal = 1./sqrt(sum(X1.*X1)+eps);
scalMat = sparse(diag(scal));
X = X1*scalMat;

D = diag(diag(L));
A = D - L;
A = abs(A); D = diag(sum(A));
P = D^-1*A;

y = sum(Y,2);
label_index = y==1;
Y = [Y, 1-y];
Ia = alpha*ones(n,1); Ia(label_index) = 0;
Ia = diag(Ia); Ib = eye(n) - Ia;
F = (eye(n) - Ia*P)\(Ib*Y);

FF = F(:,1:end-1);
B = diag(sum(FF,2));
D = diag(1./sum(FF,1));
a = sum(B(:));
Lt = B - 1/a*B*ones(n,1)*ones(1,n)*B;
Lw = B - FF*D*FF';

St = X'*Lt*X;
Sw = X'*Lw*X;
Sb = St - Sw;
score = diag(Sb)./(diag(Sw)+eps);
[score, rank] = sort(score,'descend');
1;