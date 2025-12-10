function [Xgt, tt] = helperGenerateTruthData4
    s = rng;
    rng(2019);
    % 生成三维轨迹的真实状态
    vx = 10; % m/s
    omegaA = deg2rad(5); % 转弯角速度
    omegaB = deg2rad(6); 
    omegaC = deg2rad(6);
    omegaD = deg2rad(10);
    omegaE = deg2rad(-10); % rad/s
    acc = 5; 
    accD = -3; % m/s^2
    dt = 0.1;
    tt = (0:dt:floor(3999*dt)); % 更新为适当的时间范围
    figs = [];
    
    Xgt = NaN(9, numel(tt));
    Xgt(:,1) = 0;
    
    % 定义各段的索引
    seg1 = 400;  % 匀速 1
    seg2 = 800; % 加速 1                 
    seg3 = 1200; % 旋转 1      
    seg4 = 1600; % 匀速 2
    seg5 = 2000; % 减速 1
    seg6 = 2400; % 旋转 2
    seg7 = 2800; % 匀速 3
    seg8 = 3200; % 旋转 3
    seg9 = 3600; % 加速 2
    seg10 = 4000; % 旋转 4
    seg11 = 3900; % 匀速 4
    seg12 = 4100; % 旋转 5
    seg13 = 4300; % 匀速 5
    
    % 定义姿态数组
    attitude = zeros(1, numel(tt));
    
    % 初始化速度
    Xgt(2,1) = vx;
    Xgt(5,1) = 0;
    Xgt(8,1) = 0;
    
    % 生成轨迹各段
%     匀速 1
    for m = 2:seg1
        Xgt(:,m) = Xgt(:,m-1);
        Xgt(1,m) = Xgt(1,m-1) + Xgt(2,m-1) * dt;
        Xgt(4,m) = Xgt(4,m-1) + Xgt(5,m-1) * dt;
        Xgt(7,m) = Xgt(7,m-1) + Xgt(8,m-1) * dt;
        attitude(1,m) = 1;
    end

    % 加速 1
    acc_vector = [acc; acc; acc];
    for m = seg1+1:seg2
        Xgt(:,m) = Xgt(:,m-1);
        % 更新速度
        Xgt(2,m) = Xgt(2,m-1) + acc_vector(1) * dt;
        Xgt(5,m) = Xgt(5,m-1) + acc_vector(2) * dt;
        Xgt(8,m) = Xgt(8,m-1) + acc_vector(3) * dt;
        % 更新位置
        Xgt(1,m) = Xgt(1,m-1) + Xgt(2,m) * dt;
        Xgt(4,m) = Xgt(4,m-1) + Xgt(5,m) * dt;
        Xgt(7,m) = Xgt(7,m-1) + Xgt(8,m) * dt;
        attitude(1,m) = 2;
    end
    
    % 旋转 1
    for m = seg2+1:seg3
        X0 = Xgt(:,m-1);
        omega = omegaB;
        phi = atan2(X0(5), sqrt(X0(2)^2 + X0(8)^2));
        v_total = sqrt(X0(2)^2 + X0(5)^2 + X0(8)^2);
        X0(2) = v_total * cos(phi) * cos(omega * dt * (m-seg3-1));
        X0(5) = v_total * sin(phi);
        X0(8) = v_total * cos(phi) * sin(omega * dt * (m-seg3-1));
        Xgt(1,m) = Xgt(1,m-1) + X0(2) * dt;
        Xgt(4,m) = Xgt(4,m-1) + X0(5) * dt;
        Xgt(7,m) = Xgt(7,m-1) + X0(8) * dt;
        Xgt(2,m) = X0(2);
        Xgt(5,m) = X0(5);
        Xgt(8,m) = X0(8);
        attitude(1,m) = 3;
    end

        %   匀速 2
    for m = seg3+1:seg4
        Xgt(:,m) = Xgt(:,m-1);
        Xgt(1,m) = Xgt(1,m-1) + Xgt(2,m-1) * dt;
        Xgt(4,m) = Xgt(4,m-1) + Xgt(5,m-1) * dt;
        Xgt(7,m) = Xgt(7,m-1) + Xgt(8,m-1) * dt;
        attitude(1,m) = 1;
    end

    % 减速 1
    acc_vector = [accD; accD; accD];
    for m =  seg4+1:seg5
        Xgt(:,m) = Xgt(:,m-1);
        % 更新速度
        Xgt(2,m) = Xgt(2,m-1) + acc_vector(1) * dt;
        Xgt(5,m) = Xgt(5,m-1) + acc_vector(2) * dt;
        Xgt(8,m) = Xgt(8,m-1) + acc_vector(3) * dt;
        % 更新位置
        Xgt(1,m) = Xgt(1,m-1) + Xgt(2,m) * dt;
        Xgt(4,m) = Xgt(4,m-1) + Xgt(5,m) * dt;
        Xgt(7,m) = Xgt(7,m-1) + Xgt(8,m) * dt;
        attitude(1,m) = 2;
    end



      % 旋转 2
    for m = seg5+1:seg6
        X0 = Xgt(:,m-1);
        omega = omegaC;
        phi = atan2(X0(5), sqrt(X0(2)^2 + X0(8)^2));
        v_total = sqrt(X0(2)^2 + X0(5)^2 + X0(8)^2);
        X0(2) = v_total * cos(phi) * cos(omega * dt * (m-seg3-1));
        X0(5) = v_total * sin(phi);
        X0(8) = v_total * cos(phi) * sin(omega * dt * (m-seg3-1));
        Xgt(1,m) = Xgt(1,m-1) + X0(2) * dt;
        Xgt(4,m) = Xgt(4,m-1) + X0(5) * dt;
        Xgt(7,m) = Xgt(7,m-1) + X0(8) * dt;
        Xgt(2,m) = X0(2);
        Xgt(5,m) = X0(5);
        Xgt(8,m) = X0(8);
        attitude(1,m) = 3;
    end
    
    % 匀速 2
    for m = seg6+1:seg7
        Xgt(:,m) = Xgt(:,m-1);
        Xgt(1,m) = Xgt(1,m-1) + Xgt(2,m-1) * dt;
        Xgt(4,m) = Xgt(4,m-1) + Xgt(5,m-1) * dt;
        Xgt(7,m) = Xgt(7,m-1) + Xgt(8,m-1) * dt;
        attitude(1,m) = 1;
    end
    
    % 旋转 3
    for m = seg7+1:seg8
        X0 = Xgt(:,m-1);
        omega = omegaC;
        phi = atan2(X0(5), sqrt(X0(2)^2 + X0(8)^2));
        v_total = sqrt(X0(2)^2 + X0(5)^2 + X0(8)^2);
        X0(2) = v_total * cos(phi) * cos(omega * dt * (m-seg3-1));
        X0(5) = v_total * sin(phi);
        X0(8) = v_total * cos(phi) * sin(omega * dt * (m-seg3-1));
        Xgt(1,m) = Xgt(1,m-1) + X0(2) * dt;
        Xgt(4,m) = Xgt(4,m-1) + X0(5) * dt;
        Xgt(7,m) = Xgt(7,m-1) + X0(8) * dt;
        Xgt(2,m) = X0(2);
        Xgt(5,m) = X0(5);
        Xgt(8,m) = X0(8);
        attitude(1,m) = 3;
    end

    % 加速 1
    acc_vector = [acc; acc; acc];
    for m = seg8+1:seg9
        Xgt(:,m) = Xgt(:,m-1);
        % 更新速度
        Xgt(2,m) = Xgt(2,m-1) + acc_vector(1) * dt;
        Xgt(5,m) = Xgt(5,m-1) + acc_vector(2) * dt;
        Xgt(8,m) = Xgt(8,m-1) + acc_vector(3) * dt;
        % 更新位置
        Xgt(1,m) = Xgt(1,m-1) + Xgt(2,m) * dt;
        Xgt(4,m) = Xgt(4,m-1) + Xgt(5,m) * dt;
        Xgt(7,m) = Xgt(7,m-1) + Xgt(8,m) * dt;
        attitude(1,m) = 2;
    end

     % 旋转 1，在三维空间中实现螺旋上升或下降
    for m = seg9+1:seg10
        X0 = Xgt(:,m-1);
        omega = omegaA;
        phi = atan2(X0(5), sqrt(X0(2)^2 + X0(8)^2));
        % 更新速度方向
        v_total = sqrt(X0(2)^2 + X0(5)^2 + X0(8)^2);
        X0(2) = v_total * cos(phi) * cos(omega * dt * (m-2));
        X0(5) = v_total * sin(phi);
        X0(8) = v_total * cos(phi) * sin(omega * dt * (m-2));
        % 更新位置
        Xgt(1,m) = Xgt(1,m-1) + X0(2) * dt;
        Xgt(4,m) = Xgt(4,m-1) + X0(5) * dt;
        Xgt(7,m) = Xgt(7,m-1) + X0(8) * dt;
        % 更新速度
        Xgt(2,m) = X0(2);
        Xgt(5,m) = X0(5);
        Xgt(8,m) = X0(8);
        attitude(1,m) = 3;
    end
    Xgt = Xgt([1, 2, 4, 5, 7, 8], :);

        % 添加测量噪声
    measNoiseTruePos = randn(size(Xgt))*2;
    Xgt = Xgt + measNoiseTruePos;
    % 如果需要，可以添加更多的图例项
    rng(s);
end


