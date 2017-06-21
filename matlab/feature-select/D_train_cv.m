function [teResults,mseResults] = D_train_cv(traindata,testdata,trainlabel,testlabel,opt)

%特征选择的参数start
fs=0.2;
%特征选择的参数end

for k = 1:opt.cvk
    cvind1(:,k) = crossvalind('Kfold',size(find(trainlabel == 1),1),opt.cvk);
    cvind2(:,k) = crossvalind('Kfold',size(find(trainlabel == 2),1),opt.cvk);
    cvind3(:,k) = crossvalind('Kfold',size(find(trainlabel == 3),1),opt.cvk);
    cvind4(:,k) = crossvalind('Kfold',size(find(trainlabel == 4),1),opt.cvk);
    cvind5(:,k) = crossvalind('Kfold',size(find(trainlabel == 5),1),opt.cvk);
    cvind6(:,k) = crossvalind('Kfold',size(find(trainlabel == 6),1),opt.cvk);
%     cvind7(:,k) = crossvalind('Kfold',size(find(trainlabel == 7),1),opt.cvk);
%     cvind8(:,k) = crossvalind('Kfold',size(find(trainlabel == 8),1),opt.cvk);
%     cvind9(:,k) = crossvalind('Kfold',size(find(trainlabel == 9),1),opt.cvk);
%     cvind10(:,k) = crossvalind('Kfold',size(find(trainlabel == 10),1),opt.cvk);
%      cvind11(:,k) = crossvalind('Kfold',size(find(trainlabel == 11),1),opt.cvk);
%     cvind12(:,k) = crossvalind('Kfold',size(find(trainlabel == 12),1),opt.cvk);
%     cvind13(:,k) = crossvalind('Kfold',size(find(trainlabel == 13),1),opt.cvk);
%     cvind14(:,k) = crossvalind('Kfold',size(find(trainlabel == 14),1),opt.cvk);
%     cvind15(:,k) = crossvalind('Kfold',size(find(trainlabel == 15),1),opt.cvk);
end
% cvind = [cvind1;cvind2;cvind3;cvind4;cvind5;cvind6;cvind7;cvind8;cvind9;cvind10;cvind11;cvind12;cvind13;cvind14;cvind15];
cvind = [cvind1;cvind2;cvind3;cvind4;cvind5;cvind6];
for i = 1:length(opt.lambda1)
     for j = 1:length(opt.lambda2)
%          for z = 1:length(opt.lambda3)
            for c = 1:1:length(opt.c)
               for g = 1:1:length(opt.g)
                   switch opt.libsvmType
                      case 'rbf'
                        %cmdsvr =['-t 2 -h 0 -s 3' ' -c ' num2str(2^parc(c)) ' -g ' num2str(10^parg(g))];
                        cmdsvm = ['-t 2 -h 0 -s 0' ' -c ' num2str(2^opt.c(c)) ' -g ' num2str(10^opt.g(g))];
                      case 'poly'
                        %cmdsvr=['-t 1 -h 0 -s 3' ' -c ' num2str(2^parc(c)) ' -g ' num2str(10^parg(g))];
                        cmdsvm = ['-t 1 -h 0 -s 0' ' -c ' num2str(2^opt.c(c)) ' -g ' num2str(10^opt.g(g))];
                      case 'sigm'
                        %cmdsvr=['-t 3 -h 0 -s 3' ' -c ' num2str(2^parc(c)) ' -g ' num2str(10^parg(g))];
                        cmdsvm = ['-t 3 -h 0 -s 0' ' -c ' num2str(2^opt.c(c)) ' -g ' num2str(10^opt.g(g))];
                      case 'rbflib'
                        %cmdsvm = ['-t 4 -h 0 -s 3' '-c ', num2str(2^parc(c)), ' -g ', num2str(2^parg(g))];
                        cmdsvr = ['-t 4 -h 0 -s 0' '-c ', num2str(2^opt.c(c)), ' -g ', num2str(2^opt.g(g))];
                      case 'liblinear'
                        %cmdsvr = ['-q -t 4 -s 3 -h 0 -e 0.001 -w1 1.0 -c ', num2str(2^parc(c))];
                        cmdsvm= ['-q -t 0 -s  0 -h 0 -e 0.001 -w1 1.0 -c ', num2str(2^opt.c(c))];
                end
                
                tempacc = 0;
                for k = 1:opt.cvk
                    corss = (cvind(:,k) == k);     train = ~corss;                    
%                     [W] = F2L21(traindata(train,:)',trainlabel(train,:),...
%                         opt.lambda(i), opt.u(j), opt.Ite, opt.epsilon);

                     r = 8;
%                   [S] = F2graphL21(traindata(train,:),opt.lambda1(i),opt.lambda2(j));
%                     [AB, ~, ~] = SRFS_ABS(traindata(train,:),opt.lambda1(i),opt.lambda2(j),opt.lambda3(z),opt.r);
                    
%算法调用
                    [AB, ~, ~] = xijAB_ABS(traindata(train,:),opt.lambda1(i),opt.lambda2(j),opt.r);
                    %[AB, ~, ~] = CSFS(traindata(train,:)',trainlabel(train,:),opt);
                    
%算法调用
                    S = AB;
                    % 2. feature selection
                    normW = sqrt(sum(S.*S,2));
                    normW(normW <= fs*mean(normW))=0;
                    
                    SelectFeaIdx = find(normW~=0);
                    traindatawF = traindata(:,SelectFeaIdx);           

                    
                    
                    model{k} = svmtrain(trainlabel(train,:), sparse(traindatawF(train,:)), cmdsvm);
                    [pred_label, Lacc, pred_value] = svmpredict(trainlabel(corss,:),...
                        sparse(traindatawF(corss,:)), model{k});
                    tempacc (k) = Lacc(1,1);
                    %temp_pred_value{k} = pred_value;
                end
                % save the best training parameter
                [~, maxind] = max(tempacc(:));
                % best
                %best_pred_value(c,g,:) = temp_pred_value{maxind};
                train_model{i,j,c,g} = model{maxind};                
                trainacc(i,j,c,g) = mean(tempacc);
            end
        end
    end
%   end
end
[~, trmaxind] = max(trainacc(:));
[opt.besti,opt.bestj,opt.bestc,opt.bestg] = ind2sub(size(trainacc),trmaxind);
%trResults.train_pred_value = best_pred_value(opt.bestc,opt.bestg,:);

%  test stage
clear cmdsvm
switch opt.libsvmType
    case 'rbf'
        %cmdsvr =['-t 2 -h 0 -s 3' ' -c ' num2str(2^parc(c)) ' -g ' num2str(10^parg(g))];
        cmdsvm = ['-t 2 -h 0 -s 0' ' -c ' num2str(2^opt.c(opt.bestc)) ' -g ' num2str(10^opt.g(opt.bestg))];
    case 'poly'
        %cmdsvr=['-t 1 -h 0 -s 3' ' -c ' num2str(2^parc(c)) ' -g ' num2str(10^parg(g))];
        cmdsvm = ['-t 1 -h 0 -s 0' ' -c ' num2str(2^opt.c(opt.bestc)) ' -g ' num2str(10^opt.g(opt.bestg))];
    case 'sigm'
        %cmdsvr=['-t 3 -h 0 -s 3' ' -c ' num2str(2^parc(c)) ' -g ' num2str(10^parg(g))];
        cmdsvm = ['-t 3 -h 0 -s 0' ' -c ' num2str(2^opt.c(opt.bestc)) ' -g ' num2str(10^opt.g(opt.bestg))];
    case 'rbflib'
        %cmdsvm = ['-t 4 -h 0 -s 3' '-c ', num2str(2^parc(c)), ' -g ', num2str(2^parg(g))];
        cmdsvm = ['-t 4 -h 0 -s 0' '-c ', num2str(2^opt.c(opt.bestc)), ' -g ', num2str(2^opt.g(opt.bestg))];
    case 'liblinear'
        %cmdsvr = ['-q -t 4 -s 3 -h 0 -e 0.001 -w1 1.0 -c ', num2str(2^parc(c))];
        cmdsvm= ['-q -t 0 -s  0 -h 0 -e 0.001 -w1 1.0 -c ', num2str(2^opt.c(opt.bestc))];
end

% test stage
%  feature selection
clear W normW SelectFeaIdx traindatawF

% [S] = F2graphL21(traindata,opt.lambda1(opt.besti),opt.lambda2(opt.bestj));

[AB, ~, ~] = xijAB_ABS(traindata,opt.lambda1(opt.besti),opt.lambda2(opt.bestj),opt.r);
%[AB, ~, ~] = CSFS(traindata',trainlabel,opt);


% [AB, ~, ~] = SRFS_ABS(traindata(train,:),opt.lambda1(opt.besti),opt.lambda2(opt.bestj),opt.lambda3(opt.bestz),opt.r);
S = AB;

normW = sqrt(sum(S.*S,2));
normW(normW <= fs*mean(normW))=0;
SelectFeaIdx = find(normW~=0);
traindatawF = traindata(:,SelectFeaIdx);
testdatawF = testdata(:,SelectFeaIdx);


testmodel = svmtrain(trainlabel, sparse(traindatawF), cmdsvm);
[~, temptestacc, test_pred_value] = svmpredict(testlabel, sparse(testdatawF), testmodel);
%[teResults(1),teResults(2),teResults(3),teResults(4),teResults(5),teResults(6),teResults(7)] = AccSenSpe(test_pred_value,testlabel);
 teResults = temptestacc(1,1);
 mseResults = temptestacc(2,1); 
end