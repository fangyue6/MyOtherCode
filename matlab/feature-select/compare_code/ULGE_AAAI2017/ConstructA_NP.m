% function A = ConstructA_NP(TrainData, Anchor,k)
% %d*n
% Dis = sqdist(TrainData,Anchor);
% [~,idx] = sort(Dis,2);
% [~,anchor_num] = size(Anchor);
% [~,num] = size(TrainData);
% clear TrainData;
% Dis1 = zeros(num,k+1);
% idx1 = zeros(num,k+1);
% for i = 1:num
%     id = idx(i,2:k+2);
%     di = Dis(i,id);
%     Dis1(i,:) = di;
%     idx1(i,:) = id; 
% end;
% clear Dis idx;
% A = (zeros(num,anchor_num));
% for i = 1:num
%     id = idx1(i,:);
%     di = Dis1(i,:);
%     A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
% end;
% A = sparse(A);

function A = ConstructA_NP(TrainData, Anchor,k)
%d*n
Dis = sqdist(TrainData,Anchor);
[~,idx] = sort(Dis,2);
[~,anchor_num] = size(Anchor);
[~,num] = size(TrainData);
A = zeros(num,anchor_num);
for i = 1:num
    id = idx(i,1:k+1);
    di = Dis(i,id);
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end;
A = sparse(A);