
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%初始化
h_mon = zeros(batchsize,epoch)';         %epoch行，batchsize列
H_mon = zeros(batchsize,epoch)';         %epoch行，batchsize列
theta_update1_mon = zeros(epoch+1,1);    %epoch+1行
theta_update2_mon = zeros(epoch+1,1); 
theta_update1_mon(1) = theta_1;
theta_update2_mon(1) = theta_2;
lost_mon = zeros(epoch,1);               %epoch+1行
JLost_mon = zeros(epoch,1);
ftheta1_mon = zeros(epoch,1);            %thera1的偏导数，epoch+1行
ftheta2_mon = zeros(epoch,1);            %thera2的偏导数，epoch+1行
ftheta_mon = zeros(epoch,2);
normftheta_mon = zeros(epoch,1);         %梯度的模

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%function_动量法加速%%%%%%%%%%%%%%%%%%%%%%%%%%%
gamma = 0.9;alpha_mon = 0.01;
vtheta_1_mon = zeros(epoch+1,1);
vtheta_2_mon = zeros(epoch+1,1);

for i = 1 : epoch  
    for j = 1:batchsize
        h_mon(i,j) = (theta_update1_mon(i) + theta_update2_mon(i) * x(sample_num(i,j))- y(sample_num(i,j)));      %线性模型
        H_mon(i,j) = h_mon(i,j)^2;                                             %平方
        lost_mon(i) = lost_mon(i) + H_mon(i,j);                                    %求和    
        ftheta1_mon(i) = ftheta1_mon(i) + h_mon(i,j);                              %theta1的偏导数
        ftheta2_mon(i) = ftheta2_mon(i) + h_mon(i,j) * x(sample_num(i,j));         %theta2的偏导数
    end
    JLost_mon(i) = 0.5 * lost_mon(i) / batchsize;                                          %代价函数的值
    
    ftheta_mon(i,:) = [ftheta1_mon(i) ftheta2_mon(i)];
    normftheta_mon(i) = norm(ftheta_mon(i),2);    
    if ( normftheta_mon(i) < 0.05 )
        return
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%参数更新%%%%%%%%
    
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








