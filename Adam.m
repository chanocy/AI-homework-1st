%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%初始化
h_Adam = zeros(batchsize,epoch)';         %epoch行，batchsize列
H_Adam = zeros(batchsize,epoch)';         %epoch行，batchsize列
theta_update1_Adam = zeros(epoch+1,1);    %epoch+1行
theta_update2_Adam = zeros(epoch+1,1); 
theta_update1_Adam(1) =  theta_1;
theta_update2_Adam(1) =  theta_2;
Stheta1_Adam = zeros(epoch,1);
Stheta2_Adam = zeros(epoch,1);
Vtheta1_Adam = zeros(epoch,1);
Vtheta2_Adam = zeros(epoch,1);
C_Vtheta1_Adam = zeros(epoch,1);
C_Vtheta2_Adam = zeros(epoch,1);
C_Stheta1_Adam = zeros(epoch,1);
C_Stheta2_Adam = zeros(epoch,1);
lost_Adam = zeros(epoch,1);               %epoch+1行
JLost_Adam = zeros(epoch,1);
ftheta1_Adam = zeros(epoch,1);            %thera1的偏导数，epoch+1行
ftheta2_Adam = zeros(epoch,1);            %thera2的偏导数，epoch+1行
ftheta_Adam = zeros(epoch,2);
normftheta_Adam = zeros(epoch,1);         %梯度的模
SSE = zeros(1,epoch);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%function_动量法加速%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ir = 0.001 ; epsilon = 1e-8;

for i = 1 : epoch  
    for j = 1:batchsize
        h_Adam(i,j) = (theta_update1_Adam(i) + theta_update2_Adam(i) * x(sample_num(i,j))- y(sample_num(i,j)));      %线性模型
        H_Adam(i,j) = h_Adam(i,j)^2;                                                  %平方
        lost_Adam(i) = lost_Adam(i) + H_Adam(i,j);                                    %求和    
        ftheta1_Adam(i) = ftheta1_Adam(i) + h_Adam(i,j);                              %theta1的偏导数
        ftheta2_Adam(i) = ftheta1_Adam(i) + h_Adam(i,j) * x(sample_num(i,j));         %theta2的偏导数
    end
      
    ftheta_Adam(i,:) = [ftheta1_Adam(i) ftheta2_Adam(i)];
    normftheta_Adam(i) = norm(ftheta_Adam(i),2);    
    
    for j = 1:1000
        SSE(i) = SSE(i) + (( theta_update1_Adadelta(i) + theta_update2_Adadelta(i) * x(j) - y(j) )^2)/1000;
    end
    
    if ( normftheta_Adam(i) < 0.05  )
        return
    end
    
    if ( SSE(i)< 14.4 )
        return
    end
    
    if (i>1)
    %%%%%%%%%%%%%%%%%%%%%参数更新%%%%%%%%
        Vtheta1_Adam(i) = 0.9 * Vtheta1_Adam(i-1) + 0.1 * (1/batchsize) * ftheta1_Adam(i);
        Vtheta2_Adam(i) = 0.9 * Vtheta2_Adam(i-1) + 0.1 * (1/batchsize) * ftheta2_Adam(i);
        
        Stheta1_Adam(i) = 0.999 * Stheta1_Adam(i-1) + 0.001 * ( (1/batchsize) * ftheta1_Adam(i) )^2;
        Stheta2_Adam(i) = 0.999 * Stheta2_Adam(i-1) + 0.001 * ( (1/batchsize) * ftheta2_Adam(i) )^2;
      
        C_Vtheta1_Adam(i) = Vtheta1_Adam(i)/(1 - 0.9^i);
        C_Vtheta2_Adam(i) = Vtheta2_Adam(i)/(1 - 0.9^i);
        
        C_Stheta1_Adam(i) = Stheta1_Adam(i)/(1-0.999^i);
        C_Stheta2_Adam(i) = Stheta2_Adam(i)/(1-0.999^i);
        
        theta_update1_Adam(i) = theta_update1_Adam(i-1) + Ir * C_Vtheta1_Adam(i) / ((C_Stheta1_Adam(i) + epsilon)^0.5);
        theta_update2_Adam(i) = theta_update2_Adam(i-1) + Ir * C_Vtheta2_Adam(i) / ((C_Stheta2_Adam(i) + epsilon)^0.5);
    end
end

thetacompare = [theta_update1_LBGD,theta_update2_LBGD,theta_update1_mon,theta_update2_mon,theta_update1_Nes,theta_update2_Nes,theta_update1_AdaGrad,theta_update2_AdaGrad,theta_update1_Adadelta,theta_update2_Adadelta,theta_update1_Adam,theta_update2_Adam];

% i=1:1:122;
% plot(i,loss_function(i))
% xlabel('epoch')
% ylabel('Lost')
% title('Adam')

