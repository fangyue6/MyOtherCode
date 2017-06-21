function [meanValue] = meanEvaluate_by_FY(para)
meanaCC = mean(para.aCC);
meanaRE = mean(para.aRE);
meanMSE = mean(para.MSE);
meanaRMSE = mean(para.aRMSE);
meanaRRMSE = mean(para.aRRMSE);
meanMAE = mean(para.MAE);
meanValue = [meanaCC, meanaRE, meanMSE, meanaRMSE, meanaRRMSE, meanMAE];
end