% clc;clear
% load('per_evl_test.mat');
% 
% data = Ytest;
% estimate = pre_y;
% 
% [n d] = size(Ytest);
function [per]=per_evl(data,estimate)
    [n d] = size(data);
    %% aRMSE average Root Mean Squared Error
    for i = 1:d
        for j = 1:n
            data1(j,:) = data(j,i)-estimate(j,i);
        end
        data2(:,i) = sqrt(sum(data1.^2)/n);
    end
    per.aRMSE = sum(data2)/d;

    %% aCC average Correlation Coefficient
    for i = 1:d
        for j = 1:n
            data11(j,:) = data(j,i)-mean(data(:,i));
            data12(j,:) = estimate(j,i)-mean(estimate(:,i));
        end
        data13(:,i) = sum(data11.*data12)/sqrt(sum(data11.^2).*sum(data12.^2));
    end
    per.aCC = sum(data13)/d;
end
