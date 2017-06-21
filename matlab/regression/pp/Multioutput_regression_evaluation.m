function [aCC aRMSE] = Multioutput_regression_evaluation(gth,pre)
% ACC: average CC: mean(cc of columns))
% ARMSE: average RMSE: mean(RMSE);
% clear;clc;[ACC ARMSE] = SNP_valuation(rand(5,3),rand(5,3));
% gth (pre): ins * fea

% Xiaofeng Zhu on 1/15/2016
[n d] = size(gth);
gthpre = [gth pre];
cc = corrcoef(gthpre);
grepre_cc = abs(cc(d+1:end,1:d));
aCC = mean(diag(grepre_cc));

product_gthpre = (gth - pre).^2;
product_gthpre = sqrt(sum(product_gthpre)./n);
aRMSE = mean(product_gthpre);


