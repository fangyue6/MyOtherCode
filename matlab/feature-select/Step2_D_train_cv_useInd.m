function [pred_label,teResults,mseResults] = Step2_D_train_cv_useInd(traindata,testdata,trainlabel,testlabel,opt,currentK,currentDocument)
%currentK当前k折
%currentDocument当前文档

%特征选择的参数start
%-1 3  0.1 98.6056   0.2 98.9534
%0  1  0.2 98.6025
fs=0.2;
%特征选择的参数end




%  test stage
clear cmdsvm
switch opt.libsvmType
    case 'rbf'
        %cmdsvr =['-t 2 -h 0 -s 3' ' -c ' num2str(2^parc(c)) ' -g ' num2str(10^parg(g))];
        cmdsvm = ['-t 2 -h 0 -s 0' ' -c ' num2str(2^opt.c(opt.c)) ' -g ' num2str(10^opt.g(opt.g))];
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
        cmdsvm= ['-q -t 0 -s  0 -h 0 -e 0.001 -w1 1.0 -c ', num2str(2^opt.c)];
end

% test stage
%  feature selection
clear W normW SelectFeaIdx traindatawF

% [S] = F2graphL21(traindata,opt.lambda1(opt.besti),opt.lambda2(opt.bestj));


t1=clock;

%%算法调用 start
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
        [AB, ~, ~] = xijAB_ABS(traindata,opt.lambda1,opt.lambda2,opt.r);
        S = AB;
    case 'CSFS'
        %[AB, ~, ~] = xijAB_ABS(traindata(train,:),opt.lambda1(i),opt.lambda2(j),opt.r)
        [AB, ~, ~] = CSFS(traindata',trainlabel,opt);
        S = AB;
   case 'FSRobust_ALM'
        [~,W1,~,~] = FSRobust_ALM(traindata',traindata,4,10,1,10); 
        S = W1;
   case 'jelsr'
         %  opt.classnum换为6
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
        %  opt.classnum换为6
        [W1,~,~] = RPCA_OM(traindata',opt.classnum,20);
        S = W1;
    case 'traceratioFS_unsupervised' 
        [W1,~,~] = traceratioFS_unsupervised(traindata,opt.classnum,opt.classnum,opt.lambda1);
        S = W1;
     case 'LS21'
        W = LS21(traindata, trainlabel, 100);
        S = W;
    case 'FSASL'
        [W,~,~,~] = FSASL(traindata',opt.classnum);
        S = W;
    case 'FSSI'
        para_FSSI.alpha = 1;
        para_FSSI.beta = 1;
        [W,~] = FSSI(traindata',trainlabel',para_FSSI);
        S = W;
     case 'UDFS'
        para_UDFS.k=10;
        para_UDFS.lamda=100;
        X=traindata();
        L = LocalDisAna(X', para_UDFS);%
        A = X'*L*X;
        [W, ~]=LquadR21_reg(A, opt.classnum, 10);
        S = W;
end
%%算法调用 end 



normW = sqrt(sum(S.*S,2));
normW(normW <= fs *mean(normW))=0;
SelectFeaIdx = find(normW~=0);
traindatawF = traindata(:,SelectFeaIdx);
testdatawF = testdata(:,SelectFeaIdx);


testmodel = svmtrain(trainlabel, sparse(traindatawF), cmdsvm);
[pred_label, temptestacc, test_pred_value] = svmpredict(testlabel, sparse(testdatawF), testmodel);
%[teResults(1),teResults(2),teResults(3),teResults(4),teResults(5),teResults(6),teResults(7)] = AccSenSpe(test_pred_value,testlabel);
 teResults = temptestacc(1,1);
 mseResults = temptestacc(2,1); 
 
 
 t2=clock;
disp(['训练集用时',num2str(etime(t2,t1)),'秒']);
end