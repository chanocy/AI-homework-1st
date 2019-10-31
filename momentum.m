
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%��ʼ��
h_mon = zeros(batchsize,epoch)';         %epoch�У�batchsize��
H_mon = zeros(batchsize,epoch)';         %epoch�У�batchsize��
theta_update1_mon = zeros(epoch+1,1);    %epoch+1��
theta_update2_mon = zeros(epoch+1,1); 
theta_update1_mon(1) = theta_1;
theta_update2_mon(1) = theta_2;
lost_mon = zeros(epoch,1);               %epoch+1��
JLost_mon = zeros(epoch,1);
ftheta1_mon = zeros(epoch,1);            %thera1��ƫ������epoch+1��
ftheta2_mon = zeros(epoch,1);            %thera2��ƫ������epoch+1��
ftheta_mon = zeros(epoch,2);
normftheta_mon = zeros(epoch,1);         %�ݶȵ�ģ

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%function_����������%%%%%%%%%%%%%%%%%%%%%%%%%%%
gamma = 0.9;alpha_mon = 0.01;
vtheta_1_mon = zeros(epoch+1,1);
vtheta_2_mon = zeros(epoch+1,1);

for i = 1 : epoch  
    for j = 1:batchsize
        h_mon(i,j) = (theta_update1_mon(i) + theta_update2_mon(i) * x(sample_num(i,j))- y(sample_num(i,j)));      %����ģ��
        H_mon(i,j) = h_mon(i,j)^2;                                             %ƽ��
        lost_mon(i) = lost_mon(i) + H_mon(i,j);                                    %���    
        ftheta1_mon(i) = ftheta1_mon(i) + h_mon(i,j);                              %theta1��ƫ����
        ftheta2_mon(i) = ftheta2_mon(i) + h_mon(i,j) * x(sample_num(i,j));         %theta2��ƫ����
    end
    JLost_mon(i) = 0.5 * lost_mon(i) / batchsize;                                          %���ۺ�����ֵ
    
    ftheta_mon(i,:) = [ftheta1_mon(i) ftheta2_mon(i)];
    normftheta_mon(i) = norm(ftheta_mon(i),2);    
    if ( normftheta_mon(i) < 0.05 )
        return
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%��������%%%%%%%%
    
    vtheta_1_mon(i+1) = 0.9 * vtheta_1_mon(i) + 0.1 * (1/batchsize) * ftheta1_mon(i);
    vtheta_2_mon(i+1) = 0.9 * vtheta_2_mon(i) + 0.1 * (1/batchsize) * ftheta2_mon(i);
    
    theta_update1_mon(i+1) = theta_update1_mon(i) - alpha_mon * vtheta_1_mon(i+1);
    theta_update2_mon(i+1) = theta_update2_mon(i) - alpha_mon * vtheta_2_mon(i+1);
  
    
end

thetacompare = [theta_update1_LBGD,theta_update2_LBGD,theta_update1_mon,theta_update2_mon];


% SSE = zeros(1,398);
% for  i = 1:398
%     for j = 1:1000
%         SSE(i) = SSE(i) + (( theta_update1_mon(i) + theta_update2_mon(i) * x(j) - y(j) )^2)/1000/2;
%     end
% end
%     
% i = 1:1:398;
% plot(i,SSE(i))
% xlabel('epoch');
% ylabel('Lost');
% title('Momentum');
% axis([0,1000,6,8])








