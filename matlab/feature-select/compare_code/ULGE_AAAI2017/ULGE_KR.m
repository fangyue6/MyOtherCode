function [W,sumTime] = ULGE_KR(X,projectionDim,alpha,numAnchor,numNearestAnchor,downsampling,sign)
% Input
% X:                num*dim data matrix
% projectionDim:    projection dimension
% alpha:            regulization parameter
% numAnchor:        number of anchors
% numNearestAnchor: number of nearest anchors
% downsampling:     decimation of downsampling for kmeans
% sign:             1 for kmeans anchor generation, 0 for random selection

% Output
% W:                dim*projectionDim projection matrix 
% sumTime:          running time

% Ref: Feiping Nie, Wei Zhu, and Xuelong Li. Unsupervised Large Graph Embedding. AAAI 2017.

X = double(X);%n*d
num = size(X,1);
dim = size(X,2);
X = X-repmat(mean(X),[num,1]);

sample_id = randperm(num,round(num/downsampling));
%% ------------------- 1. anchor generation ------------------
tic;
if sign == 1
    [~,~,~,locAnchor] = kmeans_fastest(X(sample_id,:)',numAnchor);
elseif sign == 0
    sample_anchor = randperm(num,numAnchor);
    locAnchor = X(sample_anchor,:)';
end;
toc;
time(1) = toc;
%% ------------------- 2. anchor based graph ------------------
tic;
Z = ConstructA_NP(X',locAnchor,numNearestAnchor);
sumZ = sum(Z);
sqrtZ = sumZ.^(-0.5);
regZ = (Z)*(diag(sqrtZ));
toc;
time(2) = toc;
%% ------------------ 3. spectral analysis ------------------
tic;
[V,S] = eigs(regZ'*regZ,projectionDim+1);
U = regZ*V*inv(S);
F = U(:,2:projectionDim+1);

ddata = (X'*X);
ddata = ddata + alpha*eye(dim);

ddata = max(ddata,ddata');
B = X'*F;
R = chol(ddata);
W = R\(R'\B);
toc;
time(3) = toc;
sumTime = sum(time);

