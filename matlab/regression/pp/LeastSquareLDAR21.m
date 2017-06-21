% function [outA, outB, outNumIter, outObj] = LeastSquareLDAR21(inX, inY, inR, inPara, inW0)
function [outA, outB] = LeastSquareLDAR21(inX, inY, SLRR_lambda, inR)
% Solve min_{A,B}  ||Y-X'*A*B||_F^2 + Lambda * ||AB||_2,1
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
inPara.Lambda=SLRR_lambda;
inPara.maxIter=30;
inPara.thresh=1e-10;
inX=inX';

% check 
[d, n1] = size(inX);
% if( d > n1 )
%     error('the feature dim is larger than the data dim !');
% end
[n2, k] = size(inY);
if((n1 ~= n2) || (k == 1))
    error('The size of the input is not compatible !');
end

% parameter setting
maxIter = inPara.maxIter;
thresh = inPara.thresh;
Lambda = inPara.Lambda;
% centerize the data and label first
num_per_class = zeros(k, 1);
Y_n = inY;
for i = 1:k
    idx{i} = find(inY(:,i) == 1);
    num_per_class(i) = length(idx{i});
    Y_n(idx{i}, i) = 1/sqrt(num_per_class(i));
end
% intialization
% if(nargin < 4)
    D2 = diag(ones(d, 1));
% else              %我们自己改过
%     W = inW0;
%     Wi2 = sqrt(sum(W.*W,2) + eps);
%     d2 = 0.5./(Wi2);
%     D2 = diag(d2);
% end
XX = inX*inX';
XY = inX*Y_n;
Sb = XY*XY';
obj = zeros(maxIter, 1);
% loop
for t = 1: maxIter
    fprintf('processing iteration %d...\n', t);
    % fix D2, update A
    St = XX + Lambda * D2;
    [V, S] = eig(St\Sb);
    [s_sorted, idx_sorted] = sort(diag(S), 'descend');
    A = V(:,idx_sorted(1:inR));
    % fix D2, A, update B
    B = (A'*St*A)\A'*XY;
    % fix A, B, update D2
    W = A*B;
    Wi2 = sqrt(sum(W.*W, 2) + eps);
    d2 = 0.5./Wi2;
    D2 = diag(d2);
    % calculate 21-Norm
    W21 = sum(Wi2);
    % calculate obj
    obj(t) = norm(Y_n-inX'*A*B, 'fro')^2 + Lambda * W21; 
    if(t > 1)
        diff = obj(t-1) - obj(t);
        if(diff < thresh)
            break;
        end
    end
end
outA = A;
outB = B;
outNumIter = t;
outObj = obj;
% debug
% figure, plot(1: length(obj), obj);
end
