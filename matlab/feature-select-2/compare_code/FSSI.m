%% sample code for Multi-task learning via low rank for feature selection

% Reference: 
% 1. Yi Yang, Zhigang Ma, Alex Hauptmann and Nicu Sebe. Feature Selection for Multimedia Analysis by Sharing Information among Multiple Tasks.  
% IEEE Transactions on Multimedia

% Reference: 
% 2. Zhigang Ma, Yi Yang, Yang Cai, Nicu Sebe, Alex Hauptmann. Knowledge Adaptation for Ad Hoc Multimedia Event Detection with Few
% Examplars. 
% ACM Multimedia 2012, (ACM MM 12). [see
% http://www.cs.cmu.edu/~yiyang/SAR.m] 


% Contact: yee.i.yang@gmail.com; kevin.z.g.ma@gmail.com.

%%
function [Wm,b] = FSSI(data,gnd,para)
% input: data.X1,X2,...,Xt : data from multiple sources; dim*num
%        gnd.Y1,Y2,...,Yt : labels for multiple data; class*num 
%        para.alpha, beta: parameters.
% output: W.W1,W2,...,Wt as multiple feature selection functions; dim*class
%         b.b1,b2,...,bt as multiple bias terms;
alpha = para.alpha;
beta = para.beta;
t = size(data,2);
In = cell(1,t);
H = cell(1,t);
e = cell(1,t);
W = cell(1,t);
b = cell(1,t);
Wm = [];
for i = 1:t
    d(i) = size(data,1);
    n(i) = size(data,2);
    c(i) = size(gnd,1);
    e = ones(n(i),1);
    In = eye(n(i));
    H = In - e*e'/n(i);
    W = rand(d(i),c(i));
    b = (gnd*e-W'*data*e)/n(i);
    Wm = [Wm W];
end
iter = 1;
obji = 1;
while 1 
    D1 = cell(1,t);
    D2 = 0.5*((Wm*Wm'+eps)^(-0.5));
    Wm = [];
    for i = 1:t
        D1 = diag(0.5./sqrt(sum(W.*W,2)+eps));
        W = (data*H*data'+alpha*D1+beta*D2)\(data*H*gnd');
        b = (gnd*e-W'*data*e)/n(i);
        Wm = [Wm W];
        obj1(i) = norm(W'*data+b*e'-gnd,'fro').^2;
        obj2(i) = sum(sqrt(sum(W.*W,2)+eps));
    end
    objective = real(sum(obj1) + alpha*sum(obj2) + beta*sum(svd(Wm)));
    cver = abs((objective-obji)/obji);
    obji = objective; 
    iter = iter+1;
    if (cver < 10^-3 && iter > 2) || iter == 20, break, end
end