clc;clear all;
warning('off')

%% 定义参数start
folderPath='D:\fangyue\algorithm\feature-select-2\';
datasetsPath=[folderPath,'datasets\'];

document = {'Parkinsons2'};
%% 循环数据集 start
for d = 1:length(document)
    fileName = [datasetsPath,char(document(d)) '.mat'];
    file = load(fileName);
     %与数据对应的类数
    classnum = length(unique(file.Y));
    if classnum==2
        file.Y(file.Y==-1)=2;
    end
    
    if size(file.X,2)>400
        file.X=file.X(:,1:400);
    end
%         if size(file.X,1)>1000
%             file.X=file.X(960:1200,:);
%             file.Y=file.Y(960:1200,:);
%         end
%     
    
    [m n]=size(file.X);
    X=full(file.X);
    X = NormalizeFea(X,0);
    clear ans info
    label=file.Y;
    Y = label;
    X = X';
    
    alpha =0.1;
    beta = 1000;
    r = min(n,label)*1;
    p = 1.5;
    
    
    

[d, n] = size(X);
c = size(Y,2); %得到矩阵Y的列数
e = ones(n,1);
H = eye(n) - e*e'/n;

flagL = 'h';
flagP = 1;
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

num_per_class = zeros(size(Y,2), 1);



YYt = Y*Y';
nn = rand(n,1);
dd = rand(d,1);
P = diag(nn); 
Q = diag(dd); 
  

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
    objs(iter,1) = W21;
    objs(iter,2) = alpha * trace(AB'*XLXt*AB);
    objs(iter,3) = beta * W2p;
    
    
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
end



% save('result/our/convergent/chess_uni.mat')



