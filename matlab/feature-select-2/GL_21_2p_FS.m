function [AB, A, B, obj] = GL_21_2p_FS(X,Y,alpha,beta,r,p)
%%  motiveation:    Graph Low-Rank Feature Selection (GLR_FS) for AD diagnosis
%              1. low-rank coefficient matrix to recover noise feature matrix
%              2. global and local preservation on feature selection
%              3. transfering binary classification to multiple-output regression for exploring the diversity of the data

% Solve min_{A,B}    ||Y - X'*A*B ||_2,1 + alpha * tr(B'A'XLX'AB) + beta * ||AB||_2,p
%                 1. B = (A'(X*P*X*A + alpha*X'*L*X*A + beta*Q*A))\A'*X*P*Y;
%                 2. max_A tr((A'*(X*P*X'*A+alpha*X'*L*X*A+beta*Q*A)\A'*X*P*Y*Y'*P*X'*A);
%                 3.
%                     P21 = sqrt(sum(Y - X*AB, 2) + eps);
%                     Pdd = (1/2)./P21;
%                     P = diag(Pdd);
%                 4. 
%                   Wi2 = sqrt(sum(AB.*AB, 2) + eps).^(2-p);
%                   Qdd = (p/2)./Wi2;
%                   Q = diag(Qdd);

% input: X:    ins * fea 
%        Y:    ins * class
%        A:    fea * r (r <= min(fea,class)
%        B:    r * class
%        e:    ins * 1, e = ones(ins,1)
%        para: parameter cell
%                alpha, beta: regularization parameter
%                r: rank
%                inPara.thresh: the stopping thresh
%                inPara.p: the control paramater
%                flagL: hypergraph ('h', graph ('g')
%                weight: the weight of hypergraph
%        obj:  objective function value        

% example:  clear;clc;[AB,A,B,obj] =  GL_21_2p_FS(rand(50,100),rand(100,50),[]);

% first version by Xiaofeng Zhu on Aug. 27 2015
% second version by Xiaofeng Zhu on Aug. 28 2015 by adding hypergraph regularizer
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all;
% load('D:\fangyue\algorithm\regression\dataset\WQ');
% % fea = data;%%%%%%%%%%%%%%%%%%%%%
% % gnd = label;%%%%%%%%%%%%%%%%%%
% fea = NormalizeFea(fea,0);
% gnd = NormalizeFea(gnd,0);
% [n, d] = size(fea);
% label = size(gnd,2);
% alpha = 10;
% beta = 0.01;
% r = min(n,label)*1;
% r=4;
% p = 1.5;
% X = fea';
% Y = gnd;




[d, n] = size(X);
c = size(Y,2); %得到矩阵Y的列数
e = ones(n,1);
H = eye(n) - e*e'/n;
%   alpha = 1;
%   beta  = 2;
%   flagL = 'h';
%   Weight = ones(n,1);
%   r = 1;
%   p = 1;
% if ~exist('para', 'var')
%     alpha = 1;
%     beta  = 2;
%     r     = min(d,c);
%     flagL = 'h';
%     p = 1.5;
%     Weight = ones(n,1);
% else
%     alpha = para.alpha;
%     beta  = para.beta;
%     r     = para.r;
%     flagL = para.flagL;
%     p = para.p;
% end
flagL = 'h';
flagP = 0;
Weight = ones(n,1);

switch flagL
    case 'g'
        options = [];
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.WeightMode = 'HeatKernel';
        options.t = 1;
        W = constructW_xf(X',options);
        W = max(W,W');
        L = diag(sum(W,2)) - W;
    case 'h'  % you can define your hyperedge with multi-modality data
        Weight = ones(n,1);
        options = [];
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.WeightMode = 'HeatKernel';
        options.t = 1;
        options.bSelfConnected = 1;
        W = constructW_xf(X',options); % W: verter * edge, incident matrix matrix but with real value rather than binary value
        
        Dv = diag(sum(W)');  % the summation along row vectors
        De = diag(sum(W,2)); % the summation along col vectors
        invDe = inv(De);
        DV2 = full(Dv)^(-0.5);
        L = eye(n) - DV2 * W * diag(Weight) * invDe * W' *DV2;   % normalized hypergraph Laplacian matrix
       %L = Dv - W * diag(Weight) * invDe * W';                  % according to Shenhua Gao's PAMI13 paper 
        
end

% centerize label first
num_per_class = zeros(size(Y,2), 1);
% Y_n = Y;
% for i = 1:size(Y,2)
%     idx{i} = find(Y(:,i) == 1);
%     num_per_class(i) = length(idx{i});
%     Y_n(idx{i}, i) = 1/sqrt(num_per_class(i));
% end
% Y = Y_n;


YYt = Y*Y';
nn = rand(n,1);
dd = rand(d,1);
P = diag(nn); 
Q = diag(dd); 


% XXt  = X*X';
% XtLX = X'*L*X;
% XY   = X*Y;
% Sb   = XY*XY';

% dd = rand(d,1);
% D = diag(dd);    

XLXt = X*L*X';

iter = 1;
obji = 1;
while 1    
    
    Sb  = X*P*YYt*P*X';
    
    % update A by fixing B and P Q
    St = X*P*X' + alpha * XLXt + beta * Q;
    [V, S] = eig(St\Sb);
    [~, idx1] = sort(diag(S), 'descend');
    
    A = V(:,idx1(1:r));
    
    % update B by fixing A and P Q
    asa=A'*St*A;
    B = asa\A'*X*P*Y;
    rank(A)
    
    % update P Q by fixing A and B    
    AB = A*B;
       % calculate 2p-Norm
    Wi2 = sqrt(sum(AB.*AB, 2) + eps).^(2-p);
    Qdd = (p/2)./Wi2;
    Q = diag(Qdd);
    W2p = sum(Wi2);
    
    
    YXtAB=Y - X'*AB;

        % calculate 21-Norm
    P21 = sqrt(sum(YXtAB.*YXtAB, 2) + eps).^(1);
    Pdd = 0.5./P21;
    P = diag(Pdd);
    W21 = sum(P21);
    
    % calculate obj
%     obj(iter) = norm(Y - X'*AB, 'fro')^2 + alpha * trace(AB'*XLXt*AB) + beta * W2p; 
    obj(iter) = W21 + alpha * trace(AB'*XLXt*AB) + beta * W2p;
    
%     objj(iter,1) = W21;
%     objj(iter,2) = trace(AB'*XLXt*AB);
%     objj(iter,3) = W2p;
%     objj(iter,4) = sum(sum(A));
%     objj(iter,5) = sum(sum(B));
    

    
    cver = abs((obj(iter)-obji)/obji);
    obji = obj(iter); 
    iter = iter + 1;
%     if iter == 5,    break,     end
    if (cver < 10^-5 && iter > 2) || iter == 20,    break,     end
end


 if flagP == 1,  plot(obj), end





