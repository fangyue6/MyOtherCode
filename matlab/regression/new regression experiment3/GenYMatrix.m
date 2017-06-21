function [YY , YYY] = GenYMatrix(Y,flag)

% Y: ins*1
% flag: 1(0-1 representation), 0:Ye's method
% output:  YY = UniY*ins;
%          YYY: cell structure UniY*ins in each cell

UniY = unique(Y);
if flag ==1
    
    YY = zeros(size(Y,1),size(UniY,1));
    for i=1:length(UniY)
        idx = find(Y == UniY(i));
        YY(idx,i) = 1;
    end
    YY = YY';
else
    YY = zeros(size(UniY,1),size(Y,1));
    for i = 1:length(UniY)
        %Y = Y(:,1);
        idx = find(Y == i-1);
        YY(i,:) = -sqrt(size(idx,1)/size(Y,1));
        YY(i,idx) = sqrt(size(Y,1)/size(idx,1))-sqrt(size(idx,1)/size(Y,1));
    end
end
for i = 1:length(UniY)
    idx = find(Y == i-1);
    YYY{i} = YY(:,idx);
end
YYY = cell2mat(YYY);
