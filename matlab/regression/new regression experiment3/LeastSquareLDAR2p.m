function [outA, outB, W, outNumIter, outObj] = LeastSquareLDAR2p(inX, inY, inR, para, inW0)
% Solve min_{A,B}  ||Y-X'*A*B||_F^2 + Lambda * ||AB||_2,p
% input: inX: d by n data matrix
%        inY: n by k label matrix
%        inPara: parameter cell
%                inPara.Lambda: regularization parameter
%                inPara.maxIter: the max number of iteration
%                inPara.thresh: the stopping thresh
%        inW0: init weight matrix (optional)
%
% output: outA: d by inR
%         outB: inR by k
%         outNumIter: scalar number of interation untill satisfy
%                     stopping criterion or reach the pre-specified max 
%                     iter number
%         outObj: objective value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xiao Cai, Chris Ding, Feiping Nie, Heng Huang. 
% On The Equivalent of Low-Rank Linear Regressions and Linear Discriminant Analysis Based Regressions. 
% The 19th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2013.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% example: clear;clc;inPara.maxIter = 20;inPara.thresh =10^-5;inPara.Lambda = 5;inPara.p = 0.5;inX = rand(50,100);inY = rand(100,10);inR = 10;
%        [outA, outB, outNumIter, outObj] = LeastSquareLDAR2p(inX, inY, inR, inPara); 
% check 

% load ('dataset\EDM');
% inX = fea;
% inY = gnd;
[n1, d] = size(inX);
% [YY , YYY] = GenYMatrix(inY,flag);
%  a = YY';

[n2, k] = size(inY);
% for i = 2:k
%     inY(:,i) = inY(:,1);
% end

if((n1 ~= n2) || (k == 1))
    error('The size of the input is not compatible !');
end

% parameter setting
%inPara=struct('Maxiter',inPara.maxIter,'Thresh',inPara.thresh,'Lambda',inPara.Lambda);
maxIter = 20;
thresh = 10^-5;
Lambda = 1000;
p = 0.5;
% centerize the data and label first
num_per_class = zeros(k, 1);
Y_n = inY;
%Y_n = YY;
for i = 1:k
    idx{i} = find(inY(:,i) == 1);
    num_per_class(i) = length(idx{i});
    Y_n(idx{i}, i) = 1/sqrt(num_per_class(i));
end
% intialization
if(nargin < 6)
    D2 = diag(ones(d, 1));
else
    W = inW0;
    d2 = (p/2)./(sqrt(sum(W.*W,2)+eps).^(2-p));
    D2 = diag(d2);
end
XX = inX'*inX;
XY = inX'*Y_n;
Sb = XY*XY';
obj = zeros(maxIter, 1);
% loop
for t = 1: maxIter
    %fprintf('processing iteration %d...\n', t);
    % fix D2, update A
    St = XX + Lambda * D2;
    [V, S] = eig(St\Sb);
    [s_sorted, idx_sorted] = sort(diag(S), 'descend');
    A = V(:,idx_sorted(1:inR));
    % fix D2, A, update B
    B = (A'*St*A)\A'*XY;
    % fix A, B, update D2
    W = A*B;
    d2 = (p/2)./(sqrt(sum(W.*W,2)+eps).^(2-p));
    D2 = diag(d2);
    % calculate obj
    obj(t) = norm(Y_n-inX*A*B, 'fro')^2 + Lambda * sum(sqrt(sum(W.*W,2)+eps).^(2-p)); 
    if(t > 1)
        diff = obj(t-1) - obj(t);
        if(diff < thresh)
            break;
        end
    end
end
outA = A;
outB = B;
W = real(A*B);
outNumIter = t;
outObj = obj;
% debug

% plot(1: length(obj), obj);

end









