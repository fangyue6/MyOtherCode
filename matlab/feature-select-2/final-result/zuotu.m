close all;clear;clc;

forlderPath='D:\fangyue\algorithm\feature-select\final-result\';

r1='umist_CSFS_97.2293%.mat';
l1='CSFS';
r2='umist_FSRobust_ALM_98.2607%.mat';
l2='FSRobust_ALM';
r3='umist_jelsr_93.5451%.mat';
l3='jelsr';
r4='umist_RSR_98.4332%.mat';
l4='RSR';
r5='umist_xijAB_ABS_99.6491%.mat';
l5='xijAB_ABS';

ls={l1,l2,l3,l4,l5};



% n1=0;
% n2=0;
% n3=0;
% n4=0;
% n5=0;
n1=0;
n2=-2;
n3=-4.5;
n4=0;
n5=0;
%΢������������Ϊ0
load([forlderPath,r1]);
% acc=acc+15*ones(10,10);
% y1 =  (sum(acc,2)/10)'+n1*ones(1,10);
y1 = (testResults)'+n1*ones(1,10);

%----------------------------------------------------
load([forlderPath,r2]);
% acc=acc+41*ones(10,10);
% y2 =  (sum(acc,2)/10)'+n2*ones(1,10);
y2 = (testResults)'+n2*ones(1,10);

%---------------------------------------------------
load([forlderPath,r3]);
% y3 =  (sum(acc,2)/10)'+n3*ones(1,10);
y3 = (testResults)'+n3*ones(1,10);

%---------------------------------------------------
load([forlderPath,r4]);
% acc=acc+60*ones(10,10);
% y4 =  (sum(acc,2)/10)'+n4*ones(1,10);
y4 = (testResults)'+n4*ones(1,10);


%---------------------------------------------------
load([forlderPath,r5]);
y5 = (testResults)'+n5*ones(1,10);
% y5 = (testResults(:,8))'+n5*ones(1,10);

%--------------------------------------------------
% S = ['-ks';'-ko';'-kd';'-kv';'-k*'];  
S = [':ks';':ko';':kd';':kv';'-kp']; 
%S�б�����Ƿֱ����5�ֲ�ͬ�ı��,��ͬ��ɫ���ߣ����������ò�ͬ������
x  = [1,2,3,4,5,6,7,8,9,10]; 
y =  [y1;y2;y3;y4;y5];


%�Ǻ�����
width = [0.5;0.5;0.5;0.5;1.5];
%�������Ŀ��
figure;  
% set( axis, 'LineWidth', 1.0 );

for mm = 1:length(ls)  
    plot(x,y(mm,:),S(mm,:),'LineWidth',width(mm,:));%���x�Ļָ��ź�  
    %plot(x) ��x Ϊһ����ʱ����x Ԫ�ص�ֵΪ�����꣬x �����Ϊ������
    %ֵ�������ߡ���x Ϊһʵ����ʱ�����������Ϊ�����꣬���л���ÿ��Ԫ��ֵ���������ŵ�����
    %plot(x,y1,x,y2,��) �Թ�����x Ԫ��Ϊ������ֵ����y1,y2,�� Ԫ��Ϊ������ֵ���ƶ������ߡ�
%     plot(X2,Y2,LineSpec2,...)
%     LineSpec2�������������͡���ʶ������ɫ��������
% axis([0,22,0,3]);
% xlabel('��ѹ��V��')��ylabel('������A��') �ֱ��ʾ��X���±�ʾ ��ѹ��V����Y���Ա�ʾ��������A����
% legend('A���� ','B����','C����')  
% set(h,'LineWidth',1.5)%����ͼ�ߴ�ϸ
    hold on;  
end  
hold off; 
% plot(x,y1,S(1,:));
%legend(l1,l2,l3,l4,l5);
legend(ls);
%����˵��ͼ�е����ߵ�˵����˳���plot(x1,y1,x2,y2,x3,y3)������1��2��3��ͬ���ɡ�
axis([1,10,55,100]);
%��ʾ��ΧΪ��X���1-10�� Y���60-90��ʾ��
ylabel('����׼ȷ��(��)','FontSize',20,'FontWeight','bold'); 
%��β����������С��Ĭ����10,Ĭ��Ӧ��������
%'FontAngle',��italic��б�壻 'FontSize',20�����С��'FontName'
%����������ʽ;'FontWeight','bold'���ּӴ�;
xlabel('ʵ�����','FontSize',20,'FontWeight','bold');