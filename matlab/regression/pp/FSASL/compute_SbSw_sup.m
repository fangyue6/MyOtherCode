function [Sb, Sw, L_b, L_w] = calculate_L(X,Y)
% X: training data each row is a data;
% Y: labels
% calculate L_b and L_w defined in traditional LDA
% Sb = X*L_b*X';
% Sw = X*L_w*X';


% ====== Initialization
data_n = size(X, 1);
class_set = unique(Y);
class_n = length(class_set);

W = zeros(data_n);
% U ��ÿһ��Ϊһ�������ı�ǩ�������������Ϊ��i�࣬���ǩ�����ĵ�i��Ԫ��Ϊ1������Ϊ0
for i = 1:class_n
    U = (Y == class_set(i)); 
    count = sum(U);	% Cardinality of each class
    index = find(U==1);
    W(index,index) = 1/count; 
end;     
L_w = eye(data_n) - W;
L_t = eye(data_n) - 1/data_n*ones(data_n,1)*ones(1,data_n);
L_b = L_t - L_w;

L_w = (L_w + L_w')/2;
L_b = (L_b + L_b')/2;

Sb = X'*L_b*X;
Sw = X'*L_w*X;

% very important!
Sb = (Sb + Sb')/2;
Sw = (Sw + Sw')/2;
