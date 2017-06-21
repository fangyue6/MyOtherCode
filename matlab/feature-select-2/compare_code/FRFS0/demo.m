%% ========================================================================
% Non-convex Regularized Self-representation for Unsupervised Feature Selection, Version 1.0
% Copyright(c) 2016 P. Zhu etl. 
% All Rights Reserved.
% 
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
% ----------------------------------------------------------------------  
% 
% Please refer to the following paper
% Pengfei Zhu, Wencheng Zhu, Weizhi Wang, et al. Non-convex Regularized Self-representation 
% for Unsupervised Feature Selection[J]// Image and Vision Computing. 2016.
%  
% bibtex
% @article{zhu2016nonconvex,
%   title={Non-convex Regularized Self-representation for Unsupervised Feature Selection},
%   author={Pengfei Zhu, Wencheng Zhu, Weizhi Wang, Wangmeng Zuo, Qinghua Hu},
%   journal={Image and Vision Computing},
%   year={2016}
% }
%   
% 
% 
% Contact: {zhupengfei}@tju.edu.cn
% ----------------------------------------------------------------------

clear;
clc;
addpath(genpath(pwd));

load('orlraws10P.mat');
%X is the data matrix  of n*d , n is the number of samples and d is the
% dimension of data.
%lambad is the parameter, balance the loss function and regularizer.
%mu is largrange parameter 

%normalize data
X = NormalizeFea(X);

sam_num=size(X,1);
feature_num=size(X,2);

%lambda need to be tuned.
lam=[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,100];
lambda = lam(1);

%largrange parameter
mu = 0.1;

%index is the ranking list, obj is the loss function.
[indx,obj] = FRFS0(X,lambda,mu);
