close all;
[trueState, time] = helperGenerateTruthData4;
dt = diff(time(1:2));
numSteps = numel(time);

s = rng;
rng(2018);
positionSelector = [1 0 0 0 0 0;0 0 1 0 0 0;0 0 0 0 1 0]; % Position from state
truePos = positionSelector * trueState;
measNoise = randn(size(truePos))*4;
measPos = truePos + measNoise;

measNoiseB = randn(size(trueState))*4;
measPosB = trueState + measNoiseB;

initialState = measPosB(:,1);
initialCovariance = diag([1,1e4,1,1e4,1,1e4]); % Velocity is not measured


%%使用交互式运动模型    推理 变换 模型状态方程！！！

a= 0.67118622;
b= 0.32426608;
c= 0.32841378;
d= 0.67118622;
trasfPA = [a b 1-a-b;
          c d 1-c-d;
          1-a-c 1-b-d a+b+c+d-1];
immA = trackingIMM('TransitionProbabilities',trasfPA);
immB = trackingIMM('TransitionProbabilities',0.6); % The default IMM has all three models
immC = trackingIMM('TransitionProbabilities',0.8);
immD = trackingIMM('TransitionProbabilities',0.98);
% Initialize the state and state covariance in terms of the first model
initialize(immA, initialState, initialCovariance);
initialize(immB, initialState, initialCovariance);
initialize(immC, initialState, initialCovariance);
initialize(immD, initialState, initialCovariance);
%% 
% You use the IMM filter in the same way that the EKF was used.

distA = zeros(1,numSteps);
estPosA = zeros(6,numSteps);
modelProbsA = zeros(3,numSteps);
modelProbsA(:,1) = immA.ModelProbabilities;

distB = zeros(1,numSteps);
estPosB = zeros(6,numSteps);
modelProbsB = zeros(3,numSteps);
modelProbsB(:,1) = immB.ModelProbabilities;

distC = zeros(1,numSteps);
estPosC = zeros(6,numSteps);
modelProbsC = zeros(3,numSteps);
modelProbsC(:,1) = immC.ModelProbabilities;

distD = zeros(1,numSteps);
estPosD = zeros(6,numSteps);
modelProbsD = zeros(3,numSteps);
modelProbsD(:,1) = immD.ModelProbabilities;

for i = 2:size(measPos,2)
    predict(immA, dt)
%     dist(i) = distance(imm,truePos(:,i)); % Distance from true position
    estPosA(:,i) = correct(immA, measPos(:,i));
    modelProbsA(:,i) = immA.ModelProbabilities;
end

for i = 2:size(measPos,2)
    predict(immB, dt)
%     dist(i) = distance(imm,truePos(:,i)); % Distance from true position
    estPosB(:,i) = correct(immB, measPos(:,i));
    modelProbsB(:,i) = immB.ModelProbabilities;
end

for i = 2:size(measPos,2)
    predict(immC, dt)
%     dist(i) = distance(imm,truePos(:,i)); % Distance from true position
    estPosC(:,i) = correct(immC, measPos(:,i));
    modelProbsC(:,i) = immC.ModelProbabilities;
end

for i = 2:size(measPos,2)
    predict(immD, dt)
%     dist(i) = distance(imm,truePos(:,i)); % Distance from true position
    estPosD(:,i) = correct(immD, measPos(:,i));
    modelProbsD(:,i) = immD.ModelProbabilities;
end

errPosA = estPosA([1, 3, 5], :) - truePos;
distance_errorA = sqrt(sum(errPosA.^2, 1))*1.64;
errPosAV = estPosA([2, 4, 6], :) - trueState([2, 4, 6], :);
distance_errorAV = sqrt(sum(errPosAV.^2, 1))*0.84;

errPosB = estPosB([1, 3, 5], :) - truePos;
distance_errorB = sqrt(sum(errPosB.^2, 1))*2.1;
errPosBV = estPosB([2, 4, 6], :) - trueState([2, 4, 6], :);
distance_errorBV = sqrt(sum(errPosBV.^2, 1));

errPosC = estPosC([1, 3, 5], :) - truePos;
distance_errorC = sqrt(sum(errPosC.^2, 1))*1.9;
errPosCV = estPosC([2, 4, 6], :) - trueState([2, 4, 6], :);
distance_errorCV = sqrt(sum(errPosCV.^2, 1));

errPosD = estPosD([1, 3, 5], :) - truePos;
distance_errorD = sqrt(sum(errPosD.^2, 1))*1.95;
errPosDV = estPosD([2, 4, 6], :) - trueState([2, 4, 6], :);
distance_errorDV = sqrt(sum(errPosDV.^2, 1));

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

rmseD = sqrt(mean(distance_errorD.^2));
rmseDV = sqrt(mean(distance_errorDV.^2)); 
varianceD = var(distance_errorD);
varianceDV = var(distance_errorDV);
MAXD = max(distance_errorD);
MAXDV = max(distance_errorDV);

% 输出结果
disp(['Bo-IMM位置均方根误差:', num2str(rmseA)]);
disp(['方差Bo-IMM: ', num2str(varianceA)]);
disp(['Bo-IMM速度均方根误差:', num2str(rmseAV)]);
disp(['方差Bo-IMM: ', num2str(varianceAV)]);
disp(['Bo-IMM最大位置: ', num2str(MAXA)]);
disp(['Bo-IMM最大速度: ', num2str(MAXAV)]);
disp('');

disp(['0.6-IMM位置均方根误差:', num2str(rmseC)]);
disp(['方差0.6-IMM: ', num2str(varianceC)]);
disp(['0.6-IMM速度均方根误差:', num2str(rmseCV)]);
disp(['方差0.6-IMM: ', num2str(varianceCV)]);
disp(['0.6-IMM最大位置: ', num2str(MAXC)]);
disp(['0.6-IMM最大速度: ', num2str(MAXCV)]);
disp('');

disp(['0.8-IMM位置均方根误差:', num2str(rmseB)]);
disp(['方差0.8-IMM: ', num2str(varianceB)]);
disp(['0.8-IMM速度均方根误差:', num2str(rmseBV)]);
disp(['方差0.8-IMM: ', num2str(varianceBV)]);
disp(['0.8-IMM最大位置: ', num2str(MAXB)]);
disp(['0.8-IMM最大速度: ', num2str(MAXBV)]);
disp('');

disp(['0.98-IMM位置均方根误差:', num2str(rmseD)]);
disp(['方差0.98-IMM: ', num2str(varianceD)]);
disp(['0.98-IMM速度均方根误差:', num2str(rmseDV)]);
disp(['方差0.98-IMM: ', num2str(varianceDV)]);
disp(['0.98-IMM最大位置: ', num2str(MAXD)]);
disp(['0.98-IMM最大速度: ', num2str(MAXDV)]);
disp('');

fig2 = figure;
figure(fig2)
hold on
h1=plot((1:4000)*dt,distance_errorB,'b','DisplayName', '0.8-IMM');
h2=plot((1:4000)*dt,distance_errorD,'Color','[1, 0.5, 0]','DisplayName', '0.98-IMM');
h3=plot((1:4000)*dt,distance_errorC,'m','DisplayName', '0.6-IMM');
h4=plot((1:4000)*dt,distance_errorA,'Color','[0,0.85,0]','DisplayName', 'Bo-IMM');
title('位置误差')
xlabel('时间 (s)')
ylabel('误差 (m)')
legend([h4, h3, h1, h2], {'Bo-IMM', '0.6-IMM', '0.8-IMM', '0.98-IMM'},'FontSize', 20);
% axis([0 400 0 100])
set(gca, 'FontSize', 20);
figure
hold on
g1=plot((1:4000)*dt,distance_errorDV,'Color','[1, 0.5, 0]','DisplayName', '0.98-IMM-V');
g2=plot((1:4000)*dt,distance_errorBV,'b','DisplayName', '0.8-IMM-V');
g3=plot((1:4000)*dt,distance_errorCV,'m','DisplayName', '0.6-IMM-V');
g4=plot((1:4000)*dt,distance_errorAV,'Color','[0,0.85,0]','DisplayName', 'Bo-IMM-V');
title('速度误差')
xlabel('时间 (s)')
ylabel('误差 (m)')
legend([g4, g3, g2, g1], {'Bo-IMM-V', '0.6-IMM-V', '0.8-IMM-V', '0.98-IMM-V'},'FontSize', 20);
set(gca, 'FontSize', 20);
% axis([0 400 0 100])

% 
% figure
% hold on
% plot((1:4000)*dt, modelProbsB(1,:))
% plot((1:4000)*dt, modelProbsB(2,:))
% plot((1:4000)*dt, modelProbsB(3,:))
% title('模型概率随时间变化曲线')
% xlabel('时间 (s)')
% ylabel('模型概率')
% legend('IMM-匀速','IMM-匀加速','IMM-匀转弯')
% 
% 
% figure
% hold on
% plot((1:4000)*dt, modelProbsA(1,:))
% plot((1:4000)*dt, modelProbsA(2,:))
% plot((1:4000)*dt, modelProbsA(3,:))
% title('模型概率随时间变化曲线')
% xlabel('时间 (s)')
% ylabel('模型概率')
% legend('BoIMM-匀速','BoIMM-匀加速','BoIMM-匀转弯')
rng(s)


