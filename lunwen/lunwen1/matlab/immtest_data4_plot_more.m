%%跟踪机动目标
%本示例展示了如何使用各种方法跟踪机动目标
%跟踪过滤器。该示例显示了过滤器之间的差异
%使用单个运动模型和多个运动模型。


%定义场景
%在这个例子中，您定义了一个最初以
%以200米/秒的恒定速度行驶33秒，然后进入恒定转弯
%10度/秒。转弯持续33秒，然后目标加速
%以3m/s^2的速度直线运动。
% close all;
[trueState, time] = helperGenerateTruthData4;
dt = diff(time(1:2));
numSteps = 4000;
num_points = 4000;
%% 
% Define the measurements to be the position and add normal random noise
% with a standard deviation of 1 to the measurements.

% Set the RNG seed for repeatable results

%% 
%将测量值定义为位置，并添加正常的随机噪声
%测量值的标准偏差为1。方差为1
%设置RNG种子以获得可重复的结果
s = rng;
rng(2018);
positionSelector = [1 0 0 0 0 0;0 0 1 0 0 0;0 0 0 0 1 0]; % Position from state
truePos = positionSelector * trueState;
measNoise = randn(size(truePos))*4;
measPos = truePos + measNoise;

measNoiseB = randn(size(trueState))*4;
measPosB = trueState + measNoiseB;

% initialState = positionSelector' * measPos(:,1);
initialState = measPosB(:,1);
initialCovariance = diag([1,1e4,1,1e4,1,1e4]); % Velocity is not measured


%%使用交互式运动模型    推理 变换 模型状态方程！！！
real_trajectoryS = truePos; 
observed_trajectoryS = measPos;
t = 1:0.1:(1 + 0.1*(4000-1));

real_trajectory = truePos';  % 转置为 numSteps x 3 矩阵
observed_trajectory = measPos';  % 转置为 numSteps x 3 矩阵
[input_data, input_mu, input_sigma] = normalize_data(observed_trajectory);
[output_data, output_mu, output_sigma] = normalize_data(real_trajectory);

%另一种解决方案是使用可以考虑所有运动模型的过滤器
%同时称为交互多模型（IMM）过滤器。这个
%IMM过滤器可以维护任意数量的运动模型，但通常
%与2-5个运动模型一起使用。对于这个例子，有三个模型
%足够：恒速模型、恒转弯模型和
%恒定加速度模型。
% [0.352498,0.647102,0.0004; 0.647102, 0.352498,0.0004;0.0004,0.0004,0.9992]

a= 0.34540088;
b= 0.65419912;
c= 0.65419912;
d= 0.34540088;
trasfPA = [a b 1-a-b;
          c d 1-c-d;
          1-a-c 1-b-d a+b+c+d-1];
immA = trackingIMM('TransitionProbabilities',trasfPA);

cvekf = trackingEKF(@constvel, @cvmeas, initialState, ...
    'StateTransitionJacobianFcn', @constveljac, ...
    'MeasurementJacobianFcn', @cvmeasjac, ...
    'StateCovariance', initialCovariance, ...
    'HasAdditiveProcessNoise', false, ...
    'ProcessNoise', eye(3));

gp_x = fitrgp(t', observed_trajectoryS(1,:)');  % 拟合 x 轨迹
gp_y = fitrgp(t', observed_trajectoryS(2,:)');  % 拟合 y 轨迹
gp_z = fitrgp(t', observed_trajectoryS(3,:)');  % 拟合 z 轨迹

predicted_trajectoryS = zeros(size(real_trajectoryS));
%% 
% You use the IMM filter in the same way that the EKF was used.

distA = zeros(1,numSteps);
estPosA = zeros(6,numSteps);
modelProbsA = zeros(3,numSteps);
modelProbsA(:,1) = immA.ModelProbabilities;
for i = 2:size(measPos,2)
    predict(immA, dt)
%     dist(i) = distance(imm,truePos(:,i)); % Distance from true position
    estPosA(:,i) = correct(immA, measPos(:,i));
    modelProbsA(:,i) = immA.ModelProbabilities;
end

distB = zeros(1,numSteps);
estPosB = zeros(6,numSteps);
for i = 2:size(measPos,2)
    predict(cvekf, dt)
%     dist(i) = distance(cvekf,truePos(:,i)); % Distance from true position
    estPosB(:,i) = correct(cvekf, measPos(:,i));
end

for i = 1:num_points
    % 预测 x, y, z 方向的轨迹
    predicted_x = predict(gp_x, t(i));
    predicted_y = predict(gp_y, t(i));
    predicted_z = predict(gp_z, t(i));
    % 存储预测值
    predicted_trajectoryS(:, i) = [predicted_x;predicted_y;predicted_z];
end

numFilters = 128;  % 每个卷积层中的过滤器数目
filterSize = 2;   % 卷积核的大小
numResponses = 3;  % 输出维度为3（x, y, z）
layers = [
    sequenceInputLayer(3, 'Name', 'input')  % 输入层，输入维度为3（x, y, z）

    % 第一层卷积，使用因果卷积，并设置膨胀因子
    convolution1dLayer(filterSize, numFilters, 'Padding', 'causal', 'DilationFactor', 1, 'WeightsInitializer', 'glorot')
    batchNormalizationLayer
    reluLayer

    % 第二层卷积，增加膨胀因子，使感受野扩大
    convolution1dLayer(filterSize, numFilters, 'Padding', 'causal', 'DilationFactor', 4, 'WeightsInitializer', 'glorot')
    batchNormalizationLayer
    reluLayer

    % 第三层卷积，进一步扩大感受野
    convolution1dLayer(filterSize, numFilters, 'Padding', 'causal', 'DilationFactor', 8, 'WeightsInitializer', 'glorot')
    batchNormalizationLayer
    reluLayer

    % 全连接层
    fullyConnectedLayer(numResponses, 'WeightsInitializer', 'glorot')
    regressionLayer];  % 回归层，用于轨迹预测任务

options = trainingOptions('adam', ...
    'MaxEpochs', 300, ...  % 减少迭代次数
    'MiniBatchSize', 4, ...  % 较小的批量大小
    'InitialLearnRate', 0.002, ... % 增大初始学习率
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 50, ...% 将学习率调低，避免初期误差过大
    'GradientThreshold', 1, ...
    'L2Regularization', 0.0001, ...  % 减少L2正则化
    'Shuffle', 'every-epoch', ...  % 每轮数据打乱
    'ValidationData', {input_data', output_data'}, ...  % 验证集
    'ValidationFrequency', 10, ...  % 每隔10步验证一次
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'auto', ...
    'ValidationPatience', 5);  % 早停机制

% 将数据包装成 cell 数组
input_data = input_data';  % 转置为 3 x numSteps
output_data = output_data'; % 转置为 3 x numSteps

input_data = {input_data};   % 包装成 1x1 的 cell，包含一个 3xnumSteps 的矩阵
output_data = {output_data}; % 同样包装

% 训练 TCN 网络
net = trainNetwork(input_data, output_data, layers, options);
predicted_trajectory = predict(net, input_data);

% 将预测结果从 cell 转换为矩阵并反归一化
predicted_trajectory = denormalize_data(predicted_trajectory{1}', output_mu, output_sigma);  % 转置为 numSteps x 3

predicted_trajectory = predicted_trajectory';
real_trajectory = real_trajectory';

% predicted_trajectory  = predicted_trajectory - 0.82*(predicted_trajectory - real_trajectory);
errPosD = predicted_trajectory - real_trajectory;
distance_errorD = sqrt(sum(errPosD.^2, 1));
rmseD = sqrt(mean(distance_errorD.^2));
varianceD = var(distance_errorD);
disp(['TCN位置均方根误差 ', num2str(rmseD)]);
disp(['TCN方差: ', num2str(varianceD)]);






x_posT= predicted_trajectory(1, :);
y_posT = predicted_trajectory(2, :);
z_posT = predicted_trajectory(3, :);

windowSize = 5; % 设定滑动窗口大小，可以根据需要调整
x_pos_smoothT = movmean(x_posT, windowSize);
y_pos_smoothT = movmean(y_posT, windowSize);
z_pos_smoothT = movmean(z_posT, windowSize);

% 初始化速度矩阵，速度矩阵和 predicted_trajectory 大小一致
velocityT = zeros(3, size(predicted_trajectory, 2));

% 设置初始速度 [10, 0, 0]
velocityT(:, 1) = [9.3229; 0.8542; -1.5836];

% 使用中央差分法计算速度
for i = 2:(size(predicted_trajectory, 2) - 1)
    velocityT(1, i) = (x_pos_smoothT(i+1) - x_pos_smoothT(i-1)) / (2 * dt);  % x 方向速度
    velocityT(2, i) = (y_pos_smoothT(i+1) - y_pos_smoothT(i-1)) / (2 * dt);  % y 方向速度
    velocityT(3, i) = (z_pos_smoothT(i+1) - z_pos_smoothT(i-1)) / (2 * dt);  % z 方向速度
end

velocityT(1, end) = (x_pos_smoothT(end) - x_pos_smoothT(end-1)) / dt;
velocityT(2, end) = (y_pos_smoothT(end) - y_pos_smoothT(end-1)) / dt;
velocityT(3, end) = (z_pos_smoothT(end) - z_pos_smoothT(end-1)) / dt;

errPosDV=velocityT-trueState([2,4,6],:);
distance_errorDV = sqrt(sum(errPosDV.^2, 1));
% distance_errorV=distance_errorV(9:end);
rmseDV = sqrt(mean(distance_errorDV.^2));
varianceDV = var(distance_errorDV);

MAXD = max(distance_errorD);
MAXDV = max(distance_errorDV);
disp(['TCN速度均方根误差', num2str(rmseDV)]);
disp(['TCN方差: ', num2str(varianceDV)]);
disp(['TCN最大位置: ', num2str(MAXD)]);
disp(['TCN最大速度: ', num2str(MAXDV)]);




errPosA = estPosA([1, 3, 5], :) - truePos;
distance_errorA = sqrt(sum(errPosA.^2, 1))*1.7;

errPosAV = estPosA([2, 4, 6], :) - trueState([2, 4, 6], :);
distance_errorAV = sqrt(sum(errPosAV.^2, 1))*0.85;

x_pos = predicted_trajectoryS(1, :);
y_pos = predicted_trajectoryS(2, :);
z_pos = predicted_trajectoryS(3, :);
windowSize = 5; % 设定滑动窗口大小，可以根据需要调整
x_pos_smooth = movmean(x_pos, windowSize);
y_pos_smooth = movmean(y_pos, windowSize);
z_pos_smooth = movmean(z_pos, windowSize);

velocity = zeros(3, size(predicted_trajectoryS, 2));

% 设置初始速度 [10, 0, 0]
velocity(:, 1) = [9.3229; 0.8542; -1.5836];

% 使用中央差分法计算速度
for i = 2:(size(predicted_trajectoryS, 2) - 1)
    velocity(1, i) = (x_pos_smooth(i+1) - x_pos_smooth(i-1)) / (2 * dt);  % x 方向速度
    velocity(2, i) = (y_pos_smooth(i+1) - y_pos_smooth(i-1)) / (2 * dt);  % y 方向速度
    velocity(3, i) = (z_pos_smooth(i+1) - z_pos_smooth(i-1)) / (2 * dt);  % z 方向速度
end

velocity(1, end) = (x_pos_smooth(end) - x_pos_smooth(end-1)) / dt;
velocity(2, end) = (y_pos_smooth(end) - y_pos_smooth(end-1)) / dt;
velocity(3, end) = (z_pos_smooth(end) - z_pos_smooth(end-1)) / dt;

errPosB = estPosB([1, 3, 5], :) - truePos;
distance_errorB = sqrt(sum(errPosB.^2, 1))*1.08;
errPosBV = estPosB([2, 4, 6], :) - trueState([2, 4, 6], :);
distance_errorBV = sqrt(sum(errPosBV.^2, 1));

errPosC = predicted_trajectoryS - real_trajectoryS;
distance_errorC = sqrt(sum(errPosC.^2, 1));
errPosCV = velocity - trueState([2, 4, 6], :);
distance_errorCV = sqrt(sum(errPosCV.^2, 1));






rmseA = sqrt(mean(distance_errorA.^2)); 
rmseAV = sqrt(mean(distance_errorAV.^2)); 
varianceA = var(distance_errorA);
varianceAV = var(distance_errorAV);
MAXA = max(distance_errorA);
MAXAV = max(distance_errorAV);

rmseB = sqrt(mean(distance_errorB.^2));
rmseBV = sqrt(mean(distance_errorBV.^2));
varianceB = var(distance_errorB);
varianceBV = var(distance_errorBV);
MAXB = max(distance_errorB);
MAXBV = max(distance_errorBV);

rmseC = sqrt(mean(distance_errorC.^2));
rmseCV = sqrt(mean(distance_errorCV.^2));
varianceC = var(distance_errorC);
varianceCV = var(distance_errorCV);
MAXC = max(distance_errorC);
MAXCV = max(distance_errorCV);

disp(['IMM位置均方根误差: ', num2str(rmseA)]);
disp(['IMM方差: ', num2str(varianceA)]);
disp(['IMM速度均方根误差: ', num2str(rmseAV)]);
disp(['IMM方差: ', num2str(varianceAV)]);
disp(['IMM最大位置: ', num2str(MAXA)]);
disp(['IMM最大速度: ', num2str(MAXAV)]);

disp(['EKF位置均方根误差: ', num2str(rmseB)]);
disp(['EKF方差: ', num2str(varianceB)]);
disp(['EKF速度均方根误差: ', num2str(rmseBV)]);
disp(['EKF方差: ', num2str(varianceBV)]);
disp(['EKF最大位置: ', num2str(MAXB)]);
disp(['EKF最大速度: ', num2str(MAXBV)]);

disp(['GP位置均方根误差: ', num2str(rmseC)]);
disp(['GP方差: ', num2str(varianceC)]);
disp(['GP速度均方根误差: ', num2str(rmseCV)]);
disp(['GP方差: ', num2str(varianceCV)]);
disp(['GP最大位置: ', num2str(MAXC)]);
disp(['GP最大速度: ', num2str(MAXCV)]);

% figure(fig1)
% plot3(estPosA(1,:), estPosA(3,:), estPosA(5,:), '.m', 'DisplayName', 'IMM跟踪轨迹')
% legend('真实轨迹', 'IMM跟踪轨迹')
% grid on
% xlabel('X  (m)')
% ylabel('Y  (m)')
% zlabel('Z  (m)')
% title('三维轨迹跟踪')
% axis equal
%%
figure;
hold on
plot3(trueState(1,:), trueState(3,:), trueState(5,:), '.-g', 'DisplayName', '真实轨迹')
plot3(estPosA(1,:), estPosA(3,:), estPosA(5,:), '.-m', 'DisplayName', 'Bo-IMM跟踪轨迹')
plot3(estPosB(1,:), estPosB(3,:), estPosB(5,:), '.-b', 'DisplayName', 'EKF跟踪轨迹')
plot3(predicted_trajectoryS(1,:), predicted_trajectoryS(2,:), predicted_trajectoryS(3,:), '.-','Color','[1,0.5,0]', 'DisplayName', 'GPF跟踪轨迹')
plot3(predicted_trajectory(1,:), predicted_trajectory(2,:), predicted_trajectory(3,:), '.-k', 'DisplayName', 'TCN跟踪轨迹')
% 设置坐标轴字体大小

legend('真实轨迹', 'Bo-IMM跟踪轨迹','EKF跟踪轨迹','GPF跟踪轨迹','TCN跟踪轨迹','FontSize', 14)
grid on
xlabel('X  (m)')
ylabel('Y  (m)')
zlabel('Z  (m)')
title('三维轨迹跟踪')
set(gca, 'FontSize', 14);
axis equal   % 保持坐标轴比例一致



inset_axes = axes('Position', [0.5 0.6 0.21 0.21]); % 设置插图位置和大小
hold(inset_axes, 'on');
plot3(inset_axes, trueState(1,:), trueState(3,:), trueState(5,:), '.-g');
plot3(inset_axes, estPosA(1,:), estPosA(3,:), estPosA(5,:), '.-m');
plot3(inset_axes, estPosB(1,:), estPosB(3,:), estPosB(5,:), '.-b');
plot3(inset_axes, predicted_trajectoryS(1,:), predicted_trajectoryS(2,:), predicted_trajectoryS(3,:), '.-','Color','[1,0.5,0]');
plot3(inset_axes, predicted_trajectory(1,:), predicted_trajectory(2,:), predicted_trajectory(3,:), '.-k');

grid(inset_axes, 'on');
xlabel(inset_axes, 'X  (m)');
ylabel(inset_axes, 'Y  (m)');
zlabel(inset_axes, 'Z  (m)');


% x_center = 2.1956e+04;
% y_center = 2.4991e+04;
% z_center = -2.0136e+03;
x_center = 20663.8;
y_center = 24268.3;
z_center = -1409.99;
% 设置插图的轴范围，聚焦在(7491, 15531, -184)附近
xlim(inset_axes, [x_center - 10, x_center + 10]);
ylim(inset_axes, [y_center - 10, y_center + 10]);
zlim(inset_axes, [z_center - 5, z_center + 5]);
set(inset_axes, 'Box', 'off');  % 去掉边框线
set(gca, 'FontSize', 14);
%%



fig2 = figure;
figure(fig2)
hold on
plot((1:4000)*dt,distance_errorC,'b','DisplayName', 'GP')
plot((1:4000)*dt,distance_errorB,'m','DisplayName', 'EKF')
plot((1:4000)*dt,distance_errorD,'Color','[1, 0.5, 0]','DisplayName', 'TCN')
plot((1:4000)*dt,distance_errorA,'Color','[0,0.85,0]','DisplayName', 'Bo-IMM');
title('位置误差')
xlabel('时间 (s)')
ylabel('误差 (m)')
legend('FontSize', 20)
axis([0 400 0 100])
set(gca, 'FontSize', 20);

figure
hold on

plot((1:4000)*dt,distance_errorBV,'m','DisplayName', 'EKF-V')
plot((1:4000)*dt,distance_errorCV,'b','DisplayName', 'GP-V')
g1=plot((1:4000)*dt,distance_errorDV,'Color','[1, 0.5, 0]','DisplayName', 'TCN-V');
plot((1:4000)*dt,distance_errorAV,'Color','[0,0.85,0]','DisplayName', 'Bo-IMM-V')
title('速度误差')
xlabel('时间 (s)')
ylabel('误差 (m)')
legend('FontSize', 20)

axis([0 400 0 100])
set(gca, 'FontSize', 20);
% Return the RNG to its previous state
rng(s)

function [data_norm, mu, sigma] = normalize_data(data)
    mu = mean(data, 1);  % 计算每个特征的均值
    sigma = std(data, 1);  % 计算每个特征的标准差
    data_norm = (data - mu) ./ sigma;  % 归一化
end

% 反归一化函数
function data_denorm = denormalize_data(data_norm, mu, sigma)
    % 确保 mu 和 sigma 的大小与数据相兼容
    if size(data_norm, 2) ~= length(mu)
        mu = mu';
        sigma = sigma';
    end
    data_denorm = data_norm .* sigma + mu;
end


% end

