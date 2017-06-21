
warning('off')

%% 定义参数start
folderPath='D:\fangyue\algorithm\feature-select-2\';
datasetsPath=[folderPath,'datasets\'];

document = {'ALLAML'};
%% 循环数据集 start
for d = 1:length(document)
    fileName = [datasetsPath,char(document(d)) '.mat'];
    file = load(fileName);
     %与数据对应的类数
    classnum = length(unique(file.Y));
    if classnum==2
        file.Y(file.Y==-1)=2;
    end
    
    if size(file.X,2)>400
        file.X=file.X(:,1:400);
    end
%         if size(file.X,1)>1000
%             file.X=file.X(960:1200,:);
%             file.Y=file.Y(960:1200,:);
%         end
    
    
    [m n]=size(file.X);
    X=full(file.X);
    X = NormalizeFea(X,0);
    clear ans info
    label=file.Y;

    pars = [];
    pars.classnum=classnum;%类别数目
    pars.r = min(m,n);
    pars.cvk = 5;  %交叉验证的次数   svm训练参数   

    pars.Ite = 20;
    pars.p=1.5;

    alphas =[0.01 0.1 1 10 100 1000];
    betas =[0.01 0.1 1 10 100 1000];
for ialpha=1:length(alphas)
    pars.alpha = alphas(ialpha);
    for ibeta=1:length(betas)
        
        pars.beta = betas(ibeta);
        
        [AB, A, B, obj] = GL_21_2p_FS(X',label,pars.alpha,pars.beta,pars.r,pars.p);
        S = AB;

        % 2. feature selection
        normW = sqrt(sum(S.*S,2));
        normW(normW <= 0.2 *mean(normW))=0;

        SelectFeaIdx = find(normW~=0);
        traindatawF = X(:,SelectFeaIdx);           

        cmdsvm= ['-q -t 0 -s  0 -h 0 -e 0.001 -w1 1.0 -c ', num2str(2^2)];
        model = svmtrain(label, sparse(traindatawF), cmdsvm);
        [pred_label, Lacc, pred_value] = svmpredict(label,sparse(traindatawF), model);
        acc(ialpha,ibeta) = Lacc(1,1);
        std(ialpha,ibeta) = Lacc(2,1); 
        
    end
end
save(['result\our\parameter_sensitive\',char(document(d)),'.mat']);
end


