function [W]=MIMLfast(train_data,train_targets)
% train_data: n*1 cells, one cell for a bag, each cell is a n_ins*d matrix
% train_targets: n*n_class, one row for a bag

%% parameters
% D=100; % dimension of the shared space
[~,D] = size(train_data{1,1}); 
norm_up=1; % norm of each vector
maxiter=10; % number of iterations
step_size=0.005;  % step size of SGD
lambda=1e-5;
% num_sub=5;% number of sub concepts
num_sub=1;% number of sub concepts
opts.norm=1;
opts.average_size=10;
opts.average_begin=0;

%% initialization
train_targets=[train_targets,2*ones(size(train_targets,1),1)];
n_class=size(train_targets,2);
m=size(train_data{1},1);
costs=1./(1:n_class);
for k=2:n_class
    costs(k)=costs(k-1)+costs(k);
end

V=normrnd(0,1/sqrt(m),D,m); % D*m
W=normrnd(0,1/sqrt(m),D,n_class*num_sub); % D*n_class
for k=1:m
    tmp1=V(:,k);
    V(:,k)=tmp1*norm_up/norm(tmp1);
end
for k=1:n_class*num_sub
    tmp1=W(:,k);
    W(:,k)=tmp1*norm_up/norm(tmp1);
end

AW=0;
AV=0;
Anum=0;
trounds=0;

%% train
for i=1:maxiter
    [W,V,AW,AV,Anum,trounds]=MIML_train(train_data,train_targets,W,V,costs,norm_up,step_size,num_sub,AW,AV,Anum,trounds,lambda,opts);
end

% %% test
% [test_outputs,test_labels]=MIML_test(test_data,AW/Anum,AV/Anum,num_sub);
