% min{W,S}  \sum_ij
% S_ij*||x_i*W-x_j*W||^2+lambda_1*||Y-XW||_F^2+lambda_2*S_ij^2+lambda_3*||W||_2,1
% s.t. S_i^T*1=1, S_ij>=0, W'*W=I
function [W, S, obj] = SRFS_WS(X,Y,lambda_1,lambda_2,lambda_3)
%[W, A, B, S, obj] = SRFS_AGL(X,d, k, r,lambda_1, lambda_2, lambda_3, islocal)

%      input:
%                 X: ins*fea
%        lambda_1,2: control paremeter
%          lambda_3: regularization parameter
%           islocal:
%                       1: only update the similarities of the k neighbor pairs, the neighbor pairs are determined by the distances in the original space
%                       0: update all the similarities

%      output:
%               W:  similarity matrix
%               S: ins * ins learned symmetric similarity matrix


%%
%   1) fix S to update W
%         we can get the min_{W}  tr(W^T*X^T*L_S*XW) + lambda_1*||Y-XAB||_F^2 + lambda_3*||W||_2,1
%
%   2) fix W to update S
%         we can get the min_{S} \sum_ij S_ij*||x_i*W-x_j*W||^2+lambda_2*S_ij^2

%% Example: [W, S, obj] = SRFS_AGL(rand(50,n),[]) %
%            [W, S, obj] = SRFS_AGL(rand(100),[])  %
%           [W, S, obj] = SRFS_AGL1(rand(100,50),[])   %  向上收敛 (datasets文件夹里面中有_uni后缀的数据集都是如此)
% load('C:\Users\admin\Desktop\experiments\datasets\ecoli_uni.mat');
% X = NormalizeFea(X,0);
[n,d] = size(X);
[n,c] = size(Y);
k = 15;
eps = 10^-5;

%     lambda_1 = 0.01;
%     lambda_2 = 10;
%     lambda_3 = 10;

% XX = L2_distance_1(X,X);


%     lambda_1 = para.lambda1;
%     lambda_2  = para.lambda2;
%     lambda_3  = para.lambda3;
%     r     = para.r;
%     flagL = para.flagL;


flagL = 'g';
Weight = ones(n,1);
islocal = 0;
flagP = 1;

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
S=W;
W=rand(d,c);
Wi = sqrt(sum(W.*W, 2) + eps);
ddi = 0.5./Wi;
D_w = diag(ddi);
iter = 1;
obji = 1;
while 1
    
    % update W by fixing S
    D_s=diag(sum(S,2));
    L_s=D_s-S;
    iter_w=1;
    obji_w = 1;
%     while 1
        [W] = OptStiefelGBB_YYQ(W, X, Y, L_s, D_w, S, lambda_1, lambda_2, lambda_3);
        Wi = sqrt(sum(W.*W, 2) + eps);
        ddi = 0.5./Wi;
        D_w = diag(ddi);
        % calculate 21-Norm
        W21 = sum(Wi);
%         obj_w(iter_w) = W'*X'*L_s*X*W + lambda_1 * norm(Y - X*W, 'fro')^2 + lambda_3 * W21;
%         cver_w = abs((obj_w(iter_w)-obji_w)/obji_w);
%         obji_w = obj_w(iter_w);
%         iter_w = iter_w + 1;
%         if (cver_w < eps && iter_w > 2) || iter_w == 20,    break,     end
%     end
%     plot(obj_w)
    %% update S by fixing W
    % using Feiping Nie's AAAI 2016 code, where using the code of optimizing A.
    XW = X * W;
    distx = L2_distance_1(XW',XW');
    %     XXXX = sum(sum(distx));
    
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
    D_s=diag(sum(S,2));
    L_s=D_s-S;
    %     TTT = sum(sum(T));
    %     SSS = sum(sum(S));
    %     L = diag(sum(S)) - S;
    %     L = (L + L') / 2;
    %     LLL = sum(sum(L));
    %        [LP, objHistory] = FSASL(X,A,B, options);
    %        L = full(LP);
    
    distxi = (S - T).^2;
    % calculate obj
    obj(iter) = sum(sum(distxi .* S)) + lambda_1 * norm(Y - X*W, 'fro')^2  +  lambda_2 * norm(S)^2 + lambda_3 * W21;
    
    obj0(iter) = sum(sum(distxi .* S));
    obj1(iter) = lambda_1 * norm(Y - X*W, 'fro')^2;
    obj2(iter) = lambda_2 * norm(S)^2;
    obj3(iter) = lambda_3 * W21;
    %     xA = sum(sum(A));
    %     xB = sum(sum(B));
    %     xAB = sum(sum(AB));
    % %     sum(sum(X-XAB))
    %     x1 = sum(sum(distxi .* S));
    %     x2 =  norm(X - X*AB, 'fro')^2;
    %     x3 =  norm(S)^2;
    %     x4 =  W21;
    cver = abs((obj(iter)-obji)/obji);
    
    obji = obj(iter);
    iter = iter + 1;
    if (cver < eps && iter > 2) || iter == 20,    break,     end
end

if flagP == 1,  plot(obj), end

