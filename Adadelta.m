%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%��ʼ��
h_Adadelta = zeros(batchsize,epoch)';         %epoch�У�batchsize��
H_Adadelta = zeros(batchsize,epoch)';         %epoch�У�batchsize��
theta_update1_Adadelta = zeros(epoch+1,1);    %epoch+1��
theta_update2_Adadelta = zeros(epoch+1,1); 
theta_update1_Adadelta(1) =  theta_1;
theta_update2_Adadelta(1) =  theta_2;
Stheta1 = zeros(epoch,1);
Stheta2 = zeros(epoch,1);
Vtheta1_Adadelta = zeros(epoch,1);
Vtheta2_Adadelta = zeros(epoch,1);
delta_1 = zeros(epoch,1);
delta_2 = zeros(epoch,1);
lost_Adadelta = zeros(epoch,1);               %epoch+1��
JLost_Adadelta = zeros(epoch,1);
ftheta1_Adadelta = zeros(epoch,1);            %thera1��ƫ������epoch+1��
ftheta2_Adadelta = zeros(epoch,1);            %thera2��ƫ������epoch+1��
ftheta_Adadelta = zeros(epoch,2);
normftheta_Adadelta = zeros(epoch,1);         %�ݶȵ�ģ

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%function_����������%%%%%%%%%%%%%%%%%%%%%%%%%%%
epsilon = 1e-6;
for i = 1 : epoch  
    for j = 1:batchsize
        h_Adadelta(i,j) = (theta_update1_Adadelta(i) + theta_update2_Adadelta(i) * x(sample_num(i,j))- y(sample_num(i,j)));      %����ģ��
        H_Adadelta(i,j) = h_Adadelta(i,j)^2;                                                      %ƽ��
        lost_Adadelta(i) = lost_Adadelta(i) + H_Adadelta(i,j);                                    %���    
        ftheta1_Adadelta(i) = ftheta1_Adadelta(i) + h_Adadelta(i,j);                              %theta1��ƫ����
        ftheta2_Adadelta(i) = ftheta1_Adadelta(i) + h_Adadelta(i,j) * x(sample_num(i,j));         %theta2��ƫ����
    end
    JLost_Adadelta(i) = 0.5 * lost_Adadelta(i) / batchsize;                                          %���ۺ�����ֵ
    ftheta_Adadelta(i,:) = [ftheta1_Adadelta(i) ftheta2_Adadelta(i)];
    normftheta_Adadelta(i) = norm(ftheta_Adadelta(i),2);    
    if ( normftheta_Adadelta(i) < 0.05 )
        return
    end
    
    if (i>1)
    %%%%%%%%%%%%%%%%%%%%%��������%%%%%%%%
        Stheta1(i) = 0.9 * Stheta1(i-1) + 0.1 * (1/batchsize) * ftheta1_Adadelta(i)^2;
        Stheta2(i) = 0.9 * Stheta2(i-1) + 0.1 * (1/batchsize) * ftheta2_Adadelta(i)^2;
    
        Vtheta1_Adadelta(i) = ((delta_1(i-1) + epsilon)^0.5) / ((Stheta1(i) + epsilon)^0.5) * (1/batchsize) * ftheta1_Adadelta(i);
        Vtheta2_Adadelta(i) = ((delta_2(i-1) + epsilon)^0.5) / ((Stheta2(i) + epsilon)^0.5) * (1/batchsize) * ftheta2_Adadelta(i);
    
        delta_1(i) = 0.9 *  delta_1(i-1) + 0.1 * Vtheta1_Adadelta(i)^2;
        delta_2(i) = 0.9 *  delta_2(i-1) + 0.1 * Vtheta2_Adadelta(i)^2;

        theta_update1_Adadelta(i) = theta_update1_Adadelta(i-1) +  Vtheta1_Adadelta(i);
        theta_update2_Adadelta(i) = theta_update2_Adadelta(i-1) +  Vtheta2_Adadelta(i);
    end
end

thetacompare = [theta_update1_LBGD,theta_update2_LBGD,theta_update1_mon,theta_update2_mon,theta_update1_Nes,theta_update2_Nes,theta_update1_AdaGrad,theta_update2_AdaGrad,theta_update1_Adadelta,theta_update2_Adadelta];


% SSE = zeros(1,3000);
% for  i = 1:3000
%     for j = 1:1000
%         SSE(i) = SSE(i) + (( theta_update1_Adadelta(i) + theta_update2_Adadelta(i) * x(j) - y(j) )^2)/1000/2;
%     end
% end
%     
% i = 1:1:3000;
% plot(i,SSE(i))
% xlabel('epoch');
% ylabel('Lost');
% title('Adadelta');




