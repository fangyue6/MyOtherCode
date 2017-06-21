% min{A,B,S}  \sum_ij S_ij*||x_i*AB-x_j*AB||^2+lambda_1*S_ij^2+lambda_2*||AB||_2,1     s.t. S_i^T*1=1, S_ij>=0
function [AB, S, obj] = xijAB_ABS(X,lambda_1,lambda_2,r)
%r是秩  
%[W, A, B, S, obj] = SRFS_AGL(X,d, k, r,lambda_1, lambda_2, islocal)lambda_1,lambda_2,r

%      input: 
%                 X: ins*fea
%        lambda_1,2: control paremeter
%          lambda_3: regularization parameter
%                 r: low-rank paremeter r <= min(n,d)
%           islocal: 
%                       1: only update the similarities of the k neighbor pairs, the neighbor pairs are determined by the distances in the original space 
%                       0: update all the similarities

%      output:           
%              AB: fea * fea similarity matrix
%               S: ins * ins learned symmetric similarity matrix


%%
%   1) fix S to update A and B
%         we can get the min_{A,B}  tr(B^T*A^T*X^T*L_S*XAB) + lambda_2*||AB||_2,1
%
%   2) fix A and B to update S
%         we can get the min_{S} \sum_ij S_ij*||x_i*AB-x_j*AB||^2+lambda_1*S_ij^2
% First vension by Rongyao Hu @10/07/2016

%% Example: [AB, S, obj] = xijAB_ABS(rand(50,n),[]) %      

%这是数据集
% load('C:\Users\sdjsj\Desktop\SRFS_AGL\datasets\COIL20');
X = NormalizeFea(X,0);
[n,d] = size(X);
k = 15;
eps = 10^-5;
% r = min(n,d)
% XX = L2_distance_1(X,X);

% if ~exist(para, 'var')
%     lambda_1 = 1;
%     lambda_2 = 10;
%     r     = min(n,d);
% 
% else
%     lambda_1 = para.lambda_1;
%     lambda_2  = para.lambda_2;
%     r     = para.r;
%     flagL = para.flagL;
% end

if (~exist('lambda_1','var'))
    lambda_1 = 1; 
end

if (~exist('lambda_2','var'))
    lambda_2 = 10;
end


flagL = 'g';%  g图  h超图
Weight = ones(n,1);%两点之间的权重
islocal = 0;
flagP = 1;   %画图用到

switch flagL
    case 'g'
        options = [];
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.WeightMode = 'HeatKernel';
        options.t = 1;
        W = constructW_xf(X,options);
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

dd = rand(d,1);
D1 = diag(dd);    
S = zeros(n,n);
iter = 1;
obji = 1;
while 1  
    XLXt = X'*L*X;
    % update A , B by fixing S
    St = XLXt + lambda_2 * D1;
    [V, SS] = eig(St);%特征分解
    [~, idx1] = sort(diag(SS), 'descend');
    A = V(:,idx1(1:r));
    B = (A'*St*A)\A';
    AB = A*B;

    ABi = sqrt(sum(AB'.*AB, 2) + eps);
    ddi = 0.5./ABi;
    D1 = diag(ddi);
    
    % calculate 21-Norm
    W21 = sum(ABi);    
    
   % update S by fixing A, B 
   %% using Feiping Nie's AAAI 2016 code, where using the code of optimizing A.
    XAB = X * AB;
    distx = L2_distance_1(XAB',XAB');

    for i=1:n
        if islocal == 1
            idxa0 = idx(i,2:k+1);
        else
            idxa0 = 1:n;
        end;
        dxi = distx(i,idxa0);
        ad = -(dxi)/(2*lambda_2);
        T(i,idxa0) = EProjSimplex_new(ad);
    end;

    S = (T+T')/2;
    L = diag(sum(S)) - S;
    L = (L + L') / 2;
    LLL = sum(sum(L));
%        [LP, objHistory] = FSASL(X,A,B, options);
%        L = full(LP);

    distxi = (S - T).^2;
    % calculate obj
    obj(iter) = sum(sum(distxi .* S)) +  lambda_1 * norm(S)^2 + lambda_2 * W21;
    cver = abs((obj(iter)-obji)/obji);
    
    obji = obj(iter); 
    iter = iter + 1;
    if (cver < eps && iter > 2) || iter == 20,    break,     end
end

if flagP == 1,  plot(obj), end

