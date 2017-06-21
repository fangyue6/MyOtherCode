% CORRESPONDENCE INFORMATION
%    This code is written by Shiming Xiang and Feiping Nie

%     Shiming Xiang :  National Laboratory of Pattern Recognition,  Institute of Automation, Academy of Sciences, Beijing 100190
%                                   Email:   smxiang@gmail.com
%     Feiping Nie:       FIT Building 3-120 Room,   Tsinghua Univeristy, Beijing, China, 100084 
%                                   Email:   feipingnie@gmail.com


%    Comments and bug reports are welcome.  Email to feipingnie@gmail.com  OR  smxiang@gmail.com

%   WORK SETTING:
%    This code has been compiled and tested by using matlab    7.0

%  For more detials, please see the manuscript:
%   Shiming Xiang, Feiping Nie, Gaofeng Meng, Chunhong Pan, and Changshui Zhang. 
%   Discriminative Least Squares Regression for Multiclass Classification and Feature Selection. 
%   IEEE Transactions on Neural Netwrok and Learning System (T-NNLS),  volumn 23, issue 11, pages 1738-1754, 2012.

%   Last Modified: Nov. 2, 2012, By Shiming Xiang



function [W, b] = least_squares_regression(X,  Y,  gamma)

% X:                                     each column is a data point
% Y:                                     each column is an target data point: such as  [0, 1, 0, ..., 0]'
% gamma:                             a positive scalar

% return 
% W and b
% here we use the following equivalent model:   y = W' x + b 

[dim, N] = size (X);
% [YY , YYY] = GenYMatrix(Y,flag);
[dim_reduced, M] = size(Y);
%[dim_reduced, N] = size(Y);

% first step,  remove the mean!
XMean = mean(X')';                                       % is a column vector
XX = X - repmat(XMean, 1, N);                    % each column is a data point

W = [];
b = [];
if dim < N
    
    % W = pinv( XX * XX' + gamma * eye(dim)) * (XX * Y');
    %  Note that the above sentence can be repalced by the following sentences. So, it is more fast.
    t0 =  XX * XX' + gamma * eye(dim);
    W = t0 \ (XX * Y');   
    
    b = Y - W' * X;     % each column is an error vector
    b = mean(b')';        % now b is a column vector
    
else
    t0 = XX' * XX + gamma * eye(N);
    W = XX * (t0 \ Y);
     
   
     b = Y - X' * W;     % each column is an error vector
     plot(b)
     b = mean(b')';        % now b is a column vector
    
end




