function [teResults,mseResults] = Step2_D_train_cv(traindata,testdata,trainlabel,testlabel,opt,currentK,currentDocument)
%currentK��ǰk��
%currentDocument��ǰ�ĵ�

%����ѡ��Ĳ���start
%-1 3  0.1 98.6056   0.2 98.9534
%0  1  0.2 98.6025
fs=0.2;
%����ѡ��Ĳ���end

for k = 1:opt.cvk
    for labelIndex = 1:opt.classnum
        cv = crossvalind('Kfold',size(find(trainlabel == labelIndex),1),opt.cvk);
        %eval()�����Ĺ��ܾ��ǽ������ڵ��ַ�����Ϊ��䲢����
        eval(['cvind',num2str(labelIndex),'(:,k)', ' =cv;']);
    end
end
cvind=[];
for labelIndex = 1:opt.classnum
    eval(['cvind',' =vertcat(cvind,cvind',num2str(labelIndex),');']);
end

% for k = 1:opt.cvk
%     cvind1(:,k) = crossvalind('Kfold',size(find(trainlabel == 1),1),opt.cvk);
%     cvind2(:,k) = crossvalind('Kfold',size(find(trainlabel == 2),1),opt.cvk);
%     cvind3(:,k) = crossvalind('Kfold',size(find(trainlabel == 3),1),opt.cvk);
%     cvind4(:,k) = crossvalind('Kfold',size(find(trainlabel == 4),1),opt.cvk);
%     cvind5(:,k) = crossvalind('Kfold',size(find(trainlabel == 5),1),opt.cvk);
%     cvind6(:,k) = crossvalind('Kfold',size(find(trainlabel == 6),1),opt.cvk);
% end
% cvind = [cvind1;cvind2;cvind3;cvind4;cvind5;cvind6];
t1=clock;
sumlength=length(opt.lambda1)*length(opt.lambda2);
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
                    
                disp([currentDocument,'  ',opt.algorithm,'  ',num2str(currentK),'��  ',num2str((i-1)*j+j),'/',num2str(sumlength),'   ',num2str((c-1)*length(opt.g)+g),'��']);
                
                tempacc = 0;
                for k = 1:opt.cvk
                    corss = (cvind(:,k) == k);     train = ~corss;                    
                    
                    %�㷨���� start
                    switch opt.algorithm
                        case 'LDA'
                            [~, mapping] = compute_mapping(traindata(train,:), 'LDA');
                            S = mapping.M; 
                        case 'LR'
                            [outA,outB] = LeastSquareLDA(traindata(train,:), trainlabel(train,:), opt.r);
                            S = outA * outB;
                        case 'RSR'
                            [S,~] = L21R21(traindata(train,:),traindata(train,:),1);
                        case 'xijAB_ABS'
                            [AB, ~, ~] = xijAB_ABS(traindata(train,:),opt.lambda1(i),opt.lambda2(j),opt.r);
                            S = AB;
                        case 'CSFS'
                            [AB, ~, ~] = CSFS(traindata(train,:)',trainlabel(train,:),opt);
                            S = AB;
                       case 'FSRobust_ALM'
                            %  opt.classnum��Ϊ6
                            [~,W1,~,~] = FSRobust_ALM(traindata(train,:)',traindata(train,:),5,1,1,5); 
                            S = W1;
                       case 'jelsr'
                           %  opt.classnum��Ϊ6
                             options = [];
                             options.NeighborMode = 'KNN';
                             options.k = 5;
                             options.WeightMode = 'HeatKernel';
                             options.t = 1;
                             W = constructW_xf(traindata(train,:),options);
                            [W1,~,~] = jelsr(traindata(train,:),W,opt.classnum,opt.lambda1,opt.lambda2);
                            S = W1;
                    case 'RPCA_OM'
                            %  opt.classnum��Ϊ6
                            [W1,~,~] = RPCA_OM(traindata(train,:)',6,20);
                            S = W1;
                    case 'traceratioFS_unsupervised' 
                            [W1,~,~] = traceratioFS_unsupervised(traindata(train,:),opt.classnum,opt.classnum,opt.lambda1);
                            S = W1;
                     case 'LS21'
                            W = LS21(traindata(train,:), trainlabel(train,:), 100);
                            S = W;
                    end
                    %�㷨���� end 
                    
                    % 2. feature selection
                    normW = sqrt(sum(S.*S,2));
                    normW(normW <= fs *mean(normW))=0;
                    
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

t2=clock;
disp(['ѵ������ʱ',num2str(etime(t2,t1)),'��']);


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


t1=clock;

%%�㷨���� start
switch opt.algorithm
    case 'LDA'
        [~, mapping] = compute_mapping(traindata, 'LDA');
        S = mapping.M;
    case 'LR'
        [outA,outB] = LeastSquareLDA(traindata, trainlabel, opt.r);
        S = outA * outB;   
     case 'RSR'
        [S,~] = L21R21(traindata,traindata,1);
    case 'xijAB_ABS'
        [AB, ~, ~] = xijAB_ABS(traindata,opt.lambda1(i),opt.lambda2(j),opt.r);
        S = AB;
    case 'CSFS'
        %[AB, ~, ~] = xijAB_ABS(traindata(train,:),opt.lambda1(i),opt.lambda2(j),opt.r)
        [AB, ~, ~] = CSFS(traindata',trainlabel,opt);
        S = AB;
   case 'FSRobust_ALM'
        [~,W1,~,~] = FSRobust_ALM(traindata',traindata,4,10,1,10); 
        S = W1;
   case 'jelsr'
         %  opt.classnum��Ϊ6
        options = [];
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.WeightMode = 'HeatKernel';
        options.t = 1;
        W = constructW_xf(traindata,options);a
        %[W1,~,~] = jelsr(traindata,W,opt.classnum,opt.lambda1,opt.lambda2);
        [W1,~,~] = jelsr(traindata,W,6,1,1);
        S = W1;
    case 'RPCA_OM'
        %  opt.classnum��Ϊ6
        [W1,~,~] = RPCA_OM(traindata',opt.classnum,20);
        S = W1;
    case 'traceratioFS_unsupervised' 
        [W1,~,~] = traceratioFS_unsupervised(traindata,opt.classnum,opt.classnum,opt.lambda1);
        S = W1;
     case 'LS21'
        W = LS21(traindata, trainlabel, 100);
        S = W;
end
%%�㷨���� end 



normW = sqrt(sum(S.*S,2));
normW(normW <= fs *mean(normW))=0;
SelectFeaIdx = find(normW~=0);
traindatawF = traindata(:,SelectFeaIdx);
testdatawF = testdata(:,SelectFeaIdx);


testmodel = svmtrain(trainlabel, sparse(traindatawF), cmdsvm);
[~, temptestacc, test_pred_value] = svmpredict(testlabel, sparse(testdatawF), testmodel);
%[teResults(1),teResults(2),teResults(3),teResults(4),teResults(5),teResults(6),teResults(7)] = AccSenSpe(test_pred_value,testlabel);
 teResults = temptestacc(1,1);
 mseResults = temptestacc(2,1); 
 
 t2=clock;
disp(['ѵ������ʱ',num2str(etime(t2,t1)),'��']);
end