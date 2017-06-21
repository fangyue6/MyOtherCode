function [W,V,AW,AV,Anum,trounds]=MIML_train(train_data,train_targets,W,V,costs,norm_up,step_size0,num_sub,AW,AV,Anum,trounds,lambda,opts)

average_begin=opts.average_begin;
average_size=opts.average_size;
[n,n_class]=size(train_targets);
tmpnums=sum(train_targets>=1,2);
train_pairs=zeros(sum(tmpnums),1);
tmpidx=0;
for i=1:n
    train_pairs(tmpidx+1:tmpidx+tmpnums(i))=i;
    tmpidx=tmpidx+tmpnums(i);
end
tmptargets=train_targets';
train_pairs=[train_pairs,mod(find(tmptargets(:)>=1),n_class)];
train_pairs(train_pairs(:,2)==0,2)=n_class;

n=size(train_pairs,1);
random_idx=randperm(n);
for i=1:n
    idx_ins=train_pairs(random_idx(i),1);
    xbag=train_data{idx_ins};
    idx_class=train_pairs(random_idx(i),2);
    if(idx_class==n_class)
        idx_irr=find(train_targets(idx_ins,:)<=0);
    else
        idx_irr=find(train_targets(idx_ins,:)~=1);
    end
    n_irr=length(idx_irr);
    
    Wy=W(:,(idx_class-1)*num_sub+1:idx_class*num_sub);
    Vbag=V*xbag;
    [fs,idx_max_class]=max(Wy'*Vbag,[],1);
    [fy,idx_max_ins1]=max(fs);
    idx_max_class=idx_max_class(idx_max_ins1);
    Wy=Wy(:,idx_max_class);
    

if(1) % two optional implementation, switch to 0 to use the matlab code
    [j,idx_pick,idx_max_pick,fyn,idx_max_ins2]=sample_max1_small(n_irr,idx_irr,W,Vbag,fy,num_sub,rand(1));
else
    for j=1:n_irr
        idx_pick=idx_irr(randi(n_irr));        
        Wyn=W(:,(idx_pick-1)*num_sub+1:idx_pick*num_sub);
        [fs,idx_max_pick]=max(Wyn'*Vbag,[],1);
        [fyn,idx_max_ins2]=max(fs);
        idx_max_pick=idx_max_pick(idx_max_ins2);
        if(fyn>fy-1)
            break;
        end
    end
end


    if(fyn>fy-1) % make a gradient step, N=j;
        step_size=step_size0/(1+lambda*trounds*step_size0);
        trounds=trounds+1;
        Wyn=W(:,(idx_pick-1)*num_sub+idx_max_pick);
        loss=costs(1,floor(n_irr/j));
        
        tmp1=Wy+step_size*loss*Vbag(:,idx_max_ins1);%V*xins1;
        if(opts.norm~=0)
        tmp3=norm(tmp1);
        if(tmp3>norm_up)
            tmp1=tmp1*norm_up/tmp3;
        end
        end
        tmp2=Wyn-step_size*loss*Vbag(:,idx_max_ins2);%V*xins2;
        if(opts.norm~=0)
        tmp3=norm(tmp2);
        if(tmp3>norm_up)
            tmp2=tmp2*norm_up/tmp3;
        end
        end
        V=V-step_size*loss*(W(:,[(idx_pick-1)*num_sub+idx_max_pick,(idx_class-1)*num_sub+idx_max_class])*[xbag(:,idx_max_ins2),-xbag(:,idx_max_ins1)]');%(Wyn*xins2'-Wy*xins1');
        W(:,(idx_class-1)*num_sub+idx_max_class)=tmp1;
        W(:,(idx_pick-1)*num_sub+idx_max_pick)=tmp2;
        if(opts.norm~=0)
        norms=DNorm2(V);
        idx_down=find(norms>norm_up);
        if(~isempty(idx_down))
            norms(norms<=norm_up)=[];
            for k=1:length(idx_down)
                V(:,idx_down(k))=V(:,idx_down(k))*norm_up./norms(k);
            end
        end
        end
    end
    if(trounds>average_begin&&mod(i,average_size)==0)
        AW=AW+W;
        AV=AV+V;
        Anum=Anum+1;
       
    end
end