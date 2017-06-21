function [W ]= subspace_J21(X,Y,alpha,beta)
% Input: X, training data, dim*num;
%        Y, training data labels, num*class;
%        para.alpha, para.beta, regularization parameters;
%        para.rd, reduced dimensionality
%
%   Web Image Annotation via Subspace-Sparsity Collaborated Feature Selection. 
%   Zhigang Ma, Feiping Nie, Yi Yang, Jasper Uijlings, and Nicu Sebe. 
%   IEEE Transactions on Multimedia (TMM) 14(4):1021-1030, 2012.
X=X';

% alpha = para.alpha;
% beta = para.beta;
% rd = para.rd;
rd=0;

[dim,num] = size(X);
class = size(Y,2);
I = eye(dim);
W = rand(dim,class);
iter = 1;
obji = 1;
while 1
    da = 0.5./sqrt(sum((X'*W-Y).*(X'*W-Y),2)+eps);
    Da = diag(da);
    
    db = 0.5./sqrt(sum(W.*W,2)+eps);
    Db = diag(db);

    M = X*Da*X'+alpha*Db+beta*I;
    C = I-beta*inv(M);
    H = inv(M)*X*Da*Y*Y'*Da*X'*inv(M);
    [eigvec eigval] = eig(C\H);
    [eigval,idx] = sort(eigval,'descend');
    Q = eigvec(:,idx(1:class-rd));
    Q = real(Q);
    W = (M-beta*Q*Q')\(X*Da*Y);
    objective(iter) = sum(sqrt(sum((X'*W-Y).*(X'*W-Y),2)+eps))+alpha*sum(sqrt(sum(W.*W,2)+eps))+beta*(norm((W-Q*Q'*W),'fro'))^2;
    cver = abs((objective(iter)-obji)/obji);
    obji = objective(iter); 
    iter = iter+1;
    if (cver < 10^-3 && iter > 2) || iter ==30, break, end
end
