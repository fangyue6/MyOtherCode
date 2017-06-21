function [AB, A, B, obj] = YYQ_regression(X,Y,para)




% load ATP7d.mat;
% X = fea;
% Y = gnd;
[n,c] = size(Y);
[n,d] = size(X);


if ~exist('para', 'var')
    alpha = 1;
    beta  = 2;
    r     = min(d,c);
    flagL = 'h';
    p = 1.5;
%     Weight = ones(n,1);
else
    alpha = para.alpha;
    beta  = para.beta;
%     r     = para.r;
    r= min(d,c);
    p = para.p;
end
   

flagL = 'h';
flagP = 0;
% Weight = ones(n,1);

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
        W = constructW_xf(X,options); % W: verter * edge, incident matrix matrix but with real value rather than binary value
        
        Dv = diag(sum(W)');  % the summation along row vectors
        De = diag(sum(W,2)); % the summation along col vectors
        invDe = inv(De);
        DV2 = full(Dv)^(-0.5);
        L = eye(n) - DV2 * W * diag(Weight) * invDe * W' *DV2;   % normalized hypergraph Laplacian matrix
       %L = Dv - W * diag(Weight) * invDe * W';                  % according to Shenhua Gao's PAMI13 paper 
        
end

iter = 1;
obji = 1;


e = ones(n,1);

A = rand(c,r);
B = rand(d,r);


dd = rand(d,1);
Q = diag(dd);

H = ones(n,n) - 1/n*e*e';

XtLX = X'*L*X;
while 1    
    
    %update b
    b = (e'*Y - e'*X*B*A')./n;
    
    %update B
    B = (X'*H*X + alpha*XtLX + beta*Q)\(X'*H*Y*A);
       % calculate 2p-Norm
    Wi2 = sqrt(sum(B.*B, 2) + eps).^(2-p);
    Qdd = (p/2)./Wi2;
    Q = diag(Qdd);
    W2p = sum(Wi2);
    
    %update A
    [A] = OptStiefelGBB_YYQ(A, B, X, Y, b, e, L, alpha);
    

    
    
    
    obj1 =  norm(Y - X*B*A' - e*b, 'fro')^2;
    obj2 = alpha * trace(A*B'*XtLX*B*A');
    obj3 = beta*W2p;

    obj(iter) = obj1 + obj2 + obj3;
    
    
%    
%     % calculate 2p-Norm
%     Wi2 = sqrt(sum(AB.*AB, 2) + eps).^(2-p);
%     Qdd = (p/2)./Wi2;
%     Q = diag(Qdd);
%     W2p = sum(Wi2);
%     
%     
%     YXtAB=Y - X'*AB;
%     % calculate 21-Norm
%     P21 = sqrt(sum(YXtAB.*YXtAB, 2) + eps).^(1);
%     Pdd = 0.5./P21;
%     P = diag(Pdd);
%     W21 = sum(P21);
%     
%     % calculate obj
% %     obj(iter) = norm(Y - X'*AB, 'fro')^2 + alpha * trace(AB'*XLXt*AB) + beta * W2p; 
%     obj(iter) = W21 + alpha * trace(AB'*XLXt*AB) + beta * W2p;
%     
% %     objj(iter,1) = W21;
% %     objj(iter,2) = trace(AB'*XLXt*AB);
% %     objj(iter,3) = W2p;
% %     objj(iter,4) = sum(sum(A));
% %     objj(iter,5) = sum(sum(B));
    

    
    cver = abs((obj(iter)-obji)/obji);
    obji = obj(iter); 
    iter = iter + 1;
%     if iter == 5,    break,     end
    if (cver < 10^-5 && iter > 2) || iter == 50,    break,     end
end
AB =B*A';


 if flagP == 1,  plot(obj), end
end





