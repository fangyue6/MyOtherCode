% objective function: |Y - XBA'|_2,p + alpha * |B|_2,p, s.t., A'A = I0

% Y: ins * class, Y should normalized
% X: ins * fea, X should centered and normalized
% B: fea * rank, r<=min(class,fea)
% A: class * rank

% optimization: i)  B by fixing A:  B = (X'D1X + alpha * Dr)\(X'*D1*Y*A)
%               ii) A by fixing B:  [U S V] = mysvd(Y'*X'B); A = UV'
% example: clear;clc;X = rand(500,100);Y = rand(500,50);alpha = 0.001;p=0.5;r=20; [B,A,obj] = RRPS(X,Y,alpha,p,r);
% first version on Oct.10, 2015 
% X = rand(500,100);Y = rand(500,50);alpha = 0.001;p=0.5;r=20;

function [OutB,OutA,OutW,obj] = RRPS(X,Y,alpha,p,r)
    
XtX = X'*X;
XtY = X'*Y;
YtX = Y'*X;

d = size(X,2);
c = size(Y,2);
B = rand(d,r);
A = rand(c,r);
dr = (p/2)./(sqrt(sum(B.*B,2)+eps).^(2-p));
Dr = diag(dr);

XBAt = X*B*A';
R = Y - XBAt;
W1 = sqrt(sum(R.*R, 2) + eps);
d1 = (p/2)./W1;
D1 = diag(d1);

iter = 1;
obji = 1;
while 1
    %%%% NOTE THAT   Update B before updating A  %%%%%%%%%%%%%%%%%%%%%%% 
    %%%% NOTE THAT   Update B before updating A  %%%%%%%%%%%%%%%%%%%%%%%
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%       
    % optimize B
    XtDX = X'*D1*X;
    XtDYA = X'*D1*Y*A;
    B = (XtDX + alpha * Dr)\XtDYA;
    dr = (p/2)./(sqrt(sum(B.*B,2)+eps).^(2-p));
    Dr = diag(dr);    

    % optimize A
    YtXB = YtX * B;
%     [U, S, V] = mySVD(YtXB);
    [U, S, V] = svd(YtXB);
    if size(V,2)< r     
        break;   
    end    
    [s_sorted, idx_sorted] = sort(diag(S), 'descend');
    A = U(:,idx_sorted(1:r))*V(:,idx_sorted(1:r))'; 
    
    XBAt = X*B*A';
    R = Y - XBAt;
    W1 = sqrt(sum(R.*R, 2)+eps);
    d1 = (p/2)./W1;
    D1 = diag(d1);
    
    % calculate function value
    obj(iter) = sum(W1) + alpha * sum(sqrt(sum(B.*B,2)+eps).^(2-p));
%     obj(iter) = sum(sqrt(sum((Y - XBAt).*(Y - XBAt),2))) + alpha * sum(sqrt(sum(B.*B,2)+eps).^(2-p));
    cver = abs((obj(iter)-obji)/obji);
    obji = obj(iter); 
    iter = iter+1;
    
    OutB = B;
    OutA = A;
    OutW = B*A';
    % judge if breaking
    if (cver < 10^-9 && iter > 2) || iter == 20,    break,     end
end
% plot(obj)
end