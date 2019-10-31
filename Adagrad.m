%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%初始化
h_AdaGrad = zeros(batchsize,epoch)';         %epoch行，batchsize列
H_AdaGrad = zeros(batchsize,epoch)';         %epoch行，batchsize列
theta_update1_AdaGrad = zeros(epoch+1,1);    %epoch+1行
theta_update2_AdaGrad = zeros(epoch+1,1); 
theta_update1_AdaGrad(1) =  theta_1;
theta_update2_AdaGrad(1) =  theta_2;
Gtheta1 = zeros(epoch,1);
Gtheta2 = zeros(epoch,1);
lost_AdaGrad = zeros(epoch,1);               %epoch+1行
JLost_AdaGrad = zeros(epoch,1);
ftheta1_AdaGrad = zeros(epoch,1);            %thera1的偏导数，epoch+1行
ftheta2_AdaGrad = zeros(epoch,1);            %thera2的偏导数，epoch+1行
ftheta_AdaGrad = zeros(epoch,2);
normftheta_AdaGrad = zeros(epoch,1);         %梯度的模

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%function_动量法加速%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ir = 0.02; epsilon = 1e-7;

for i = 1 : epoch  
    for j = 1:batchsize
        h_AdaGrad(i,j) = (theta_update1_AdaGrad(i) + theta_update2_AdaGrad(i) * x(sample_num(i,j))- y(sample_num(i,j)));      %线性模型
        H_AdaGrad(i,j) = h_AdaGrad(i,j)^2;                                                     %平方
        lost_AdaGrad(i) = lost_AdaGrad(i) + H_AdaGrad(i,j);                                    %求和    
        ftheta1_AdaGrad(i) = ftheta1_AdaGrad(i) + h_AdaGrad(i,j);                              %theta1的偏导数
        ftheta2_AdaGrad(i) = ftheta2_AdaGrad(i) + h_AdaGrad(i,j) * x(sample_num(i,j));         %theta2的偏导数
    end
    JLost_AdaGrad(i) = 0.5 * lost_AdaGrad(i) / batchsize;                                      %代价函数的值
    
    ftheta_AdaGrad(i,:) = [ftheta1_AdaGrad(i) ftheta2_AdaGrad(i)];
    normftheta_AdaGrad(i) = norm(ftheta_AdaGrad(i),2);    
    if ( normftheta_AdaGrad(i) < 0.05 )
        return
    end
        
    %%%%%%%%%%%%%%%%%%%%%参数更新%%%%%%%%
    if (i > 1)
        Gtheta1(i) = Gtheta1(i-1) + (1/batchsize) * ftheta1_AdaGrad(i)^2;
        Gtheta2(i) = Gtheta2(i-1) + (1/batchsize) * ftheta2_AdaGrad(i)^2;
        theta_update1_AdaGrad(i) = theta_update1_AdaGrad(i-1) + (Ir / (Gtheta1(i)^0.5 + epsilon)) * (1/batchsize) * ftheta1_AdaGrad(i);
        theta_update2_AdaGrad(i) = theta_update2_AdaGrad(i-1) + (Ir / (Gtheta2(i)^0.5 + epsilon)) * (1/batchsize) * ftheta2_AdaGrad(i);
    end
end

thetacompare = [theta_update1_LBGD,theta_update2_LBGD,theta_update1_mon,theta_update2_mon,theta_update1_Nes,theta_update2_Nes,theta_update1_AdaGrad,theta_update2_AdaGrad];
% 
% SSE = zeros(1,3000);
% for  i = 1:3000
%     for j = 1:1000
%         SSE(i) = SSE(i) + (( theta_update1_AdaGrad(i) + theta_update2_AdaGrad(i) * x(j) - y(j) )^2)/1000/2;
%     end
% end
%     
% i = 1:1:3000;
% plot(i,SSE(i))
% xlabel('epoch');
% ylabel('Lost');
% title('AdaGrad');



