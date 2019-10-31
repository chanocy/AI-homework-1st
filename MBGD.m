%%%%%%%%%%%%%%%%%%%%��һ�δ���ҵ%%%%%%%%%%%%%%%%%%%%%%%%%
% clear 
% load('a1data.mat')
% % load('sample_num')
% p = polyfit(x,y,1)';
% 
% % maxx = ceil(max(x));
% % maxy = ceil(max(y));
% % minx = floor(min(x));
% % miny = floor(min(y));
% hold on
% y1 = polyval(p,x);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%ģ�ͻ�����������%%%%%%%%%%%%%%%%%%%%%%
% %���ó�ʼtheta
% theta_1 = 2;
% theta_2 = 1;
%���õ�������epoch��������batchsize,���������sample_num
epoch = 3000;
batchsize = 64;
sample_num = zeros(epoch,batchsize);
for i = 1:epoch
    sample_num(i,:) = randperm(1000,batchsize);      %���ѡ������
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%function_LBGD%%%%%%%%%%%%%%%%%%%%%%
h_LBGD = zeros(batchsize,epoch)';         %epoch�У�batchsize��
H_LBGD = zeros(batchsize,epoch)';         %epoch�У�batchsize��
theta_update1_LBGD = zeros(epoch+1,1);    %epoch+1��
theta_update2_LBGD = zeros(epoch+1,1);    %epoch+1��

theta_update1_LBGD(1) =  theta_1;
theta_update2_LBGD(1) =  theta_2;

lost_LBGD = zeros(epoch,1);               %epoch+1��
JLost_LBGD = zeros(epoch,1);              %epoch+1��


ftheta1_LBGD = zeros(epoch,1);            %thera1��ƫ������epoch+1��
ftheta2_LBGD = zeros(epoch,1);            %thera2��ƫ������epoch+1��
ftheta_LBGD = zeros(epoch,2);
normftheta_LBGD = zeros(epoch,1);         %�ݶȵ�ģ

%function_LBGD
alpha_LBGD = 0.005;
for i = 1 : epoch  
    for j = 1:batchsize
        h_LBGD(i,j) = (theta_update1_LBGD(i) + theta_update2_LBGD(i) * x(sample_num(i,j))- y(sample_num(i,j)));      %����ģ��
        H_LBGD(i,j) = h_LBGD(i,j)^2;                                         %ƽ��
        lost_LBGD(i) = lost_LBGD(i) + H_LBGD(i,j);                           %���    
        ftheta1_LBGD(i) = ftheta1_LBGD(i) + h_LBGD(i,j);               %theta1��ƫ����
        ftheta2_LBGD(i) = ftheta1_LBGD(i) + h_LBGD(i,j) * x(sample_num(i,j));         %theta2��ƫ����   
    end
    JLost_LBGD(i) = 0.5 * lost_LBGD(i) / batchsize ;   %���ۺ�����ֵ 
    ftheta_LBGD(i,:) = [ftheta1_LBGD(i) ftheta2_LBGD(i)];
    normftheta_LBGD(i) = norm(ftheta_LBGD(i),2);
    
    if ( normftheta_LBGD(i) < 0.01 )
        return
    end
    
    %%%%%%%%%%%%%%%%%%%%%��������%%%%%%%%
    theta_update1_LBGD(i+1) = theta_update1_LBGD(i) - alpha_LBGD * (1/batchsize) * ftheta1_LBGD(i);
    theta_update2_LBGD(i+1) = theta_update2_LBGD(i) - alpha_LBGD * (1/batchsize) * ftheta2_LBGD(i);
    
    
end
thetacompare = [theta_update1_LBGD,theta_update2_LBGD];

% SSE = zeros(1,2378);
% for  i = 1:2378
%     for j = 1:1000
%         SSE(i) = SSE(i) + (( theta_update1_LBGD(i) + theta_update2_LBGD(i) * x(j) - y(j) )^2)/1000/2;
%     end
% end
%     
% i = 1:1:2378;
% plot(i,SSE(i))
% xlabel('epoch');
% ylabel('Lost');
% title('MBGD');
% axis([0,3000,6,8])


