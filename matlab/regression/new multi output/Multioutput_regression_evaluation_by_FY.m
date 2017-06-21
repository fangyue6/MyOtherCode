function [aCC, aRE, MSE, aRMSE, aRRMSE, MAE] = Multioutput_regression_evaluation_by_FY(gth,pre)
% ACC: average CC: mean(cc of columns))
% ARMSE: average RMSE: mean(RMSE);
% clear;clc;[ACC ARMSE] = SNP_valuation(rand(5,3),rand(5,3));
% gth (pre): ins * fea

% Xiaofeng Zhu on 1/15/2016
[n d] = size(gth);%’Ê µ
gthpre = [gth pre];%‘§≤‚
cc = corrcoef(gthpre);
grepre_cc = abs(cc(d+1:end,1:d));
aCC = mean(diag(grepre_cc));

product_gthpre = (gth - pre).^2;
product_gthpre = sqrt(sum(product_gthpre)./n);
aRMSE = mean(product_gthpre);


% Yue Fang on 2016/12/10
nsum = abs(gth-pre)./gth;
aRE = sum(sum(nsum,1)/n,2)/d;%The average relative error

MSE = sum(sum(product_gthpre,1)./n,2);%The mean squared error (MSE)


sum_gth_pre_2 = sum((gth - pre).^2,1);
sum_gth_meangth_2 = sum((gth - ones(n,1)*mean(gth,1)).^2,1);
aRRMSE = sum(sqrt(sum_gth_pre_2./sum_gth_meangth_2),2)/d;%The average relative root mean squared error (aRRMSE)

MAE = sum(sum(abs(gth-ones(n,1)*mean(pre,1))))/n;%mean absolute error (MAE)

% sum(sum(abs(gth-ones(n,1)*mean(pre,1))./gth))/n;%the mean relative error (MRE)







