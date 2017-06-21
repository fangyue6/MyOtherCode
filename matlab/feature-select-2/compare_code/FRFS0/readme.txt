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

X is the data matrix  of n*d , n is the number of samples and d is the dimension of data.
\lambad is the parameter, balance the loss function and regularizer.
\mu is largrange parameter 

The demo.m is an example of our algorithm and can be run directly.
The file data contains nine datasets and the detailed information can be obtained in our paper.

