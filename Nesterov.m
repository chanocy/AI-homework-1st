%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%��ʼ��
h_Nes = zeros(batchsize,epoch)';         %epoch�У�batchsize��
H_Nes = zeros(batchsize,epoch)';         %epoch�У�batchsize��
theta_update1_Nes = zeros(epoch+1,1);    %epoch+1��
theta_update2_Nes = zeros(epoch+1,1); 
theta_update1_Nes(1) =  theta_1;
theta_update2_Nes(1) =  theta_2;
lost_Nes = zeros(epoch,1);               %epoch+1��
JLost_Nes = zeros(epoch,1);
ftheta1_Nes = zeros(epoch,1);            %thera1��ƫ������epoch+1��
ftheta2_Nes = zeros(epoch,1);            %thera2��ƫ������epoch+1��
ftheta_Nes = zeros(epoch,2);
normftheta_Nes = zeros(epoch,1);         %�ݶȵ�ģ

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%function_����������%%%%%%%%%%%%%%%%%%%%%%%%%%%
gamma = 0.9;  Ir = 0.001;
vtheta_1_Nes = zeros(epoch+1,1);
vtheta_2_Nes = zeros(epoch+1,1);

for i = 1 : epoch  
    for j = 1:batchsize
        h_Nes(i,j) = (theta_update1_Nes(i) + theta_update2_Nes(i) * x(sample_num(i,j))- y(sample_num(i,j)));      %����ģ��
        H_Nes(i,j) = h_Nes(i,j)^2;                                                 %ƽ��
        lost_Nes(i) = lost_Nes(i) + H_Nes(i,j);                                    %���    
        ftheta1_Nes(i) = ftheta1_Nes(i) + h_Nes(i,j);                              %theta1��ƫ����
        ftheta2_Nes(i) = ftheta2_Nes(i) + h_Nes(i,j) * x(sample_num(i,j));         %theta2��ƫ����
    end
    JLost_Nes(i) = 0.5 * lost_Nes(i) / batchsize;                                          %���ۺ�����ֵ
    ftheta_Nes(i,:) = [ftheta1_Nes(i) ftheta2_Nes(i)];
    normftheta_Nes(i) = norm(ftheta_Nes(i),2);    
    if ( normftheta_Nes(i) < 0.05 )
        return
    end
    
    %%%%%%%%%%%%%%%%%%%%%��������%%%%%%%%
    
    vtheta_1_Nes(i+1) = 0.9 * vtheta_1_Nes(i) -  Ir * (1/batchsize) * ftheta1_Nes(i);      %%%%i or i+1
    vtheta_2_Nes(i+1) = 0.9 * vtheta_2_Nes(i) -  Ir * (1/batchsize) * ftheta2_Nes(i);
    
    theta_update1_Nes(i+1) = theta_update1_Nes(i) + 0.9 * vtheta_1_Nes(i+1) -  Ir * (1/batchsize) * ftheta1_Nes(i);
    theta_update2_Nes(i+1) = theta_update2_Nes(i) + 0.9 * vtheta_2_Nes(i+1) -  Ir * (1/batchsize) * ftheta2_Nes(i);
  
   
end

thetacompare = [theta_update1_LBGD,theta_update2_LBGD,theta_update1_mon,theta_update2_mon,theta_update1_Nes,theta_update2_Nes];

% SSE = zeros(1,614);
% for  i = 1:614
%     for j = 1:1000
%         SSE(i) = SSE(i) + (( theta_update1_Nes(i) + theta_update2_Nes(i) * x(j) - y(j) )^2)/1000/2;
%     end
% end
%     
% i = 1:1:614;
% plot(i,SSE(i))
% plot(i,SSE(i))
% xlabel('epoch');
% ylabel('Lost');
% title('Nesterov');
% 











