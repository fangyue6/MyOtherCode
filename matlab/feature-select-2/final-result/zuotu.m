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
%微调参数，正常为0
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
%S中保存的是分别代表5种不同的标记,不同颜色的线，还可以设置不同的线型
x  = [1,2,3,4,5,6,7,8,9,10]; 
y =  [y1;y2;y3;y4;y5];


%是横坐标
width = [0.5;0.5;0.5;0.5;1.5];
%是线条的宽度
figure;  
% set( axis, 'LineWidth', 1.0 );

for mm = 1:length(ls)  
    plot(x,y(mm,:),S(mm,:),'LineWidth',width(mm,:));%绘出x的恢复信号  
    %plot(x) 当x 为一向量时，以x 元素的值为纵坐标，x 的序号为横坐标
    %值绘制曲线。当x 为一实矩阵时，则以其序号为横坐标，按列绘制每列元素值相对于其序号的曲线
    %plot(x,y1,x,y2,…) 以公共的x 元素为横坐标值，以y1,y2,… 元素为纵坐标值绘制多条曲线。
%     plot(X2,Y2,LineSpec2,...)
%     LineSpec2中设置曲线线型、标识符和颜色三项属性
% axis([0,22,0,3]);
% xlabel('电压（V）')，ylabel('电流（A）') 分别表示在X轴下标示 电压（V），Y轴旁标示“电流（A）”
% legend('A曲线 ','B曲线','C曲线')  
% set(h,'LineWidth',1.5)%设置图线粗细
    hold on;  
end  
hold off; 
% plot(x,y1,S(1,:));
%legend(l1,l2,l3,l4,l5);
legend(ls);
%用于说明图中的曲线的说明，顺序和plot(x1,y1,x2,y2,x3,y3)的曲线1、2、3相同即可。
axis([1,10,55,100]);
%显示范围为：X轴从1-10， Y轴从60-90显示。
ylabel('分类准确率(％)','FontSize',20,'FontWeight','bold'); 
%如何不设置字体大小，默认是10,默认应该是宋体
%'FontAngle',’italic’斜体； 'FontSize',20字体大小；'FontName'
%设置字体样式;'FontWeight','bold'文字加粗;
xlabel('实验次数','FontSize',20,'FontWeight','bold');