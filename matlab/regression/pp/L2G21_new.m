% function [W b obj eTime] = L2G21_new(inX, inY, A, r, MaxNumIter, objThresh)
function [W ] = L2G21_new(inX, inY, r)
% solve the following problem,
% obj = min_W{0.5||Y-X^TW||_F^2 + sum_<i,j>{A_<i,j>*||w_i, w_j||_2,1
% where w_i and w_j are the i-th and j-th column of W
% input:
%       X: d by n
%       inY: n by c
%       A: consine class-wise similarity matrix, c by c 
%           (using a pre-defined threshold to remove the smaller
%           pair-wise similarity)
%       r: regularization parameter
% output:
%       W: d by c
%       b: c by 1
%       obj: obj function value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% centerize data
 MaxNumIter=30;
A=cosine_similarity(inY);
objThresh=10^-9;
inX=inX';

[d, n] = size(inX);
H = eye(n) - (1/n)*(ones(n, 1)* ones(1,n));
X = inX*H;
Y = H*inY;
t1 = cputime;
dim = size(X,1);
c = size(Y,2);
% init W
W = (X*X'+r*eye(dim))\(X*Y);
for iter = 1:MaxNumIter
    for i = 1:c
        % update Di
        ai = A(i,:);
        idx = find(ai>0);        
        Di = zeros(dim, dim);
        for j = 1:length(idx)
            Wi = [W(:,i), W(:,idx(j))];
            di = sqrt(sum(Wi.*Wi,2)+eps);
            Di = Di + ai(idx(j))*diag(1./(2*di));
        end;
        aa = A(:,i);
        idx1 = find(ai>0);
        for j = 1:length(idx1)
            Wi = [W(:,i), W(:,idx1(j))];
            di = sqrt(sum(Wi.*Wi,2)+eps);
            Di = Di + aa(idx(j))*diag(1./(2*di));
        end;        
        % update W
        W(:,i) = (X*X'+r*Di)\(X*Y(:,i));
    end;
    % calculate b
    b = (1/n)*inY'*ones(n, 1)-(1/n)*W'*inX*ones(n,1);
    % calculate the obj value
    w21 = zeros(1, c);
    for i = 1:c
        ai = A(i,:);
        idx = find(ai>0);    
        wi21 = zeros(1, c);
        for j = 1:length(idx)
            Wi = [W(:,i), W(:,idx(j))];
            di = sqrt(sum(Wi.*Wi,2)+eps);
            wi21(j) = ai(idx(j))*sum(di);
        end;
        w21(i) = sum(wi21);
    end;
    obj(iter) = trace((X'*W-Y)'*(X'*W-Y)) + r*sum(w21);

    if(iter > 1)
        if((obj(iter -1) - obj(iter)) < objThresh)
            break;
        end
    end
end;
eTime = cputime -t1;
% debug
% figure, plot(obj);
end