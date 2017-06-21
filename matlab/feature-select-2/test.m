% %%定义参数start
% folderPath='C:\Users\fangyue\Desktop\numberone\number4\';
% datasetsPath=[folderPath,'datasets\'];
% document = {'solar_uni','umist','chess_uni'};
% classnum=[]
% %%定义参数end
% 
% %与数据对应的类数
% for d = 1:length(document)
%     fileName = [datasetsPath,char(document(d)) '.mat'];
%     file = load(fileName);
%     classnum = [classnum length(unique(file.Y))];
% end

% load('datasets/PCMAC.mat');
% [n,m] = size(X);
% x=X(1:500,1:500);
for i=1:10
    eval(['t_',num2str(i)])
end

