% min_{W'W=I, F, G \in Ind} ||W'*X-F*G'||^2/trace(W'*St*W) + r*||W||_21;
function [W, fidx, obj] = traceratioFS_unsupervised(X, m, k, r)
% X: training data, each row is a data point
% m: the reduced dimensionality
% k: number of clusters, can be set k=m
% r: parameter
% W: projection matrix
% fidx: the sorted feature index
% obj: objective value


[n,dim] = size(X);
NITER = 1;
ismax = 0;
obj = zeros(50,1);

D = eye(dim);
W = orth(rand(dim,m));
for iter = 1:50
    for it = 1:10
    label0 = ceil(k*rand(1,n));
    [Y(:,it),sumd,center] = kmeans_ldj(X*W,label0);
    obkm(it) = sum(sumd);
    end;
    [~, idx] = min(obkm);
    [Sb, Sw] = calculate_L(X,Y(:,idx));
    St = Sb+Sw;
    [W, ob] = tracesumratio(Sw,St,D,m,ismax,NITER,W);
    
    wi = sqrt(sum(W.*W,2)+eps);
    d = 0.5./wi;
    D = r*diag(d);
    obj(iter) = trace(W'*Sw*W)/trace(W'*St*W) + r*sum(wi);
end;

[temp, fidx] = sort(wi,'descend');







% max_{W'W=I} trace(W'*A*W)/trace(W'*B*W)+trace(W'*C*W);
function [W obj] = tracesumratio(A,B,C,k,ismax,NITER,W)

d = size(A,1);
%W = eye(d,k);
%W = orth(rand(d,k));
%W = eig1(B,k);
lambda = trace(W'*A*W)/trace(W'*B*W)+trace(W'*C*W);
obj(1) = lambda;
for iter = 1:NITER
    lam = trace(W'*A*W)/trace(W'*B*W);
    [W obj0] = eig1(A-lam*B+trace(W'*B*W)*C,k,ismax);
    lambda = trace(W'*A*W)/trace(W'*B*W)+trace(W'*C*W);
    
    obj(iter+1) = lambda;
end;




% max_{W'W=I} trace(W'*A*W)/trace(W'*B*W)+trace(W'*C*W);
function [W obj] = tracesumratio1(A,B,C,k,ismax,NITER,W)

d = size(A,1);
%W = eye(d,k);
%W = orth(rand(d,k));
lambda = trace(W'*A*W)/trace(W'*B*W)+trace(W'*C*W);
obj(1) = lambda;
for iter = 1:NITER
    [W obj0] = tracesummultiply(A-lambda*B,B,C,k,ismax,W);
    lambda = trace(W'*A*W)/trace(W'*B*W)+trace(W'*C*W);
    
    obj(iter+1) = lambda;
end;


% max_{W'W=I} trace(W'*A*W)+trace(W'*B*W)*trace(W'*C*W);
function [W obj] = tracesummultiply(A,B,C,k,ismax, W)

d = size(A,1);
%W = eye(d,k);
%W = orth(rand(d,k));
obj(1) = trace(W'*A*W)+trace(W'*B*W)*trace(W'*C*W);
for iter = 1:10
    M = A+trace(W'*B*W)*C+trace(W'*C*W)*B;
    W = eig1(M,k,ismax);
    
    obj(iter+1) = trace(W'*A*W)+trace(W'*B*W)*trace(W'*C*W);
end;



