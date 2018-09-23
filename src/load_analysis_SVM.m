%% author: mrinmoy sarkar
% email: mrinmoy.pol@gmail.com
% Modified and Tested by Dhiman Chowdhury
% Power (Load/Demand) Data: Dhaka, Chittagong and Rajshahi
%%
clear all;
close all;
warning('off','all');
% listing = dir('..\data\');
% fileNames = listing(3:8);
% path = listing.folder;
% trainDataset=zeros(468,4);
% indx = 1;
% for i=1:length(fileNames)
%     filename = strcat(path,'\',fileNames(i).name);
%     data = xlsread(filename);
%     traindata = data(6,1:26);
%     for j=1:length(traindata)
%         trainDataset(indx,1) = 1; % 1 means dhaka
%         trainDataset(indx,2) = i; % day
%         trainDataset(indx,3) = j; % time
%         trainDataset(indx,4) = traindata(j); % load
%         indx = indx+1;
%     end
%     traindata = data(7,1:26);
%     for j=1:length(traindata)
%         trainDataset(indx,1) = 2; % 2 means Chittagong
%         trainDataset(indx,2) = i; % day
%         trainDataset(indx,3) = j; % time
%         trainDataset(indx,4) = traindata(j); % load
%         indx = indx+1;
%     end
%     traindata = data(9,1:26);
%     for j=1:length(traindata)
%         trainDataset(indx,1) = 3; % 3 means Rajshahi
%         trainDataset(indx,2) = i; % day
%         trainDataset(indx,3) = j; % time
%         trainDataset(indx,4) = traindata(j); % load
%         indx = indx+1;
%     end
% end
% save('train.mat','trainDataset')
load trainv2.mat
noOftrial = 1;
training_error_NoPrune = zeros(1,noOftrial);
prediction_error_NoPrune = zeros(1,noOftrial);
MAE_prediction_NoPrune = zeros(1,noOftrial);
training_error_prune = zeros(1,noOftrial);
prediction_error_prune = zeros(1,noOftrial);
MAE_prediction_prune = zeros(1,noOftrial);
time1=zeros(1,noOftrial); % elapsed time without pruning
time2=zeros(1,noOftrial); % elapsed time with pruning
for trial=1:noOftrial
shuffletrainData = randperm(size(trainDataset,1));
trainSamples = fix(0.8 * length(shuffletrainData)); % 80% trained data and 20% tested data: standard practice
X = trainDataset(shuffletrainData(1:trainSamples),1:3);
Y = trainDataset(shuffletrainData(1:trainSamples),end);
X_test = trainDataset(shuffletrainData(trainSamples+1:end),1:3);
Y_test = trainDataset(shuffletrainData(trainSamples+1:end),end);
%% without pruning
tic
rtree = fitrsvm(X,Y,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));%svm
time1(trial) = toc;
%fprintf('elapsed time_without pruning: %.6f \n', time1);
%uncomment to view the trained tree (should be viewed and attached to paper)
%view(rtree,'mode','graph');

%mean squared error
totaltrainingloss1 = loss(rtree,X,Y)/length(Y);
training_error_NoPrune(trial) = totaltrainingloss1;
fprintf('total error in training: %.2f \n', totaltrainingloss1);

Y_est = predict(rtree,X_test);

%rms prediction error
predicterror1 = (sum((Y_test-Y_est).^2)/length(Y_test)).^0.5;
prediction_error_NoPrune(trial) = predicterror1;
fprintf('RMS prediction error: %.2f \n', predicterror1);

%mean absolute error
mae_prediction1 = sum(abs(Y_test-Y_est))/length(Y_test);
MAE_prediction_NoPrune(trial) = mae_prediction1;
fprintf('MAE prediction error: %.2f \n', mae_prediction1);
%% with pruning
% pruning cost value (alpha) is subject to change within [0,1]
% alpha = 0.5; 
% Level = fix(alpha*max(rtree.PruneList));
% tic
% rtreepruned = prune(rtree,'Level',Level);
% time2(trial) = toc;
% %fprintf('elapsed time_with pruning: %.6f \n', time2);
% 
% % uncomment to view the trained tree (should be viewed and attached to paper)
% %view(rtreepruned,'mode','graph'); 
% 
% %mean squared error
% totaltrainingloss2 = loss(rtreepruned, X, Y)/length(Y);
% training_error_prune(trial) = totaltrainingloss2;
% fprintf('total error in training with pruning: %.2f \n', totaltrainingloss2);
% 
% Y_est = predict(rtreepruned, X_test);
% 
% % rms prediction error
% predicterror2 = (sum((Y_test-Y_est).^2)/length(Y_test)).^0.5;
% prediction_error_prune(trial) = predicterror2;
% fprintf('RMS prediction error with pruning: %.2f \n', predicterror2);
% 
% %mean absolute error
% mae_prediction2 = sum(abs(Y_test-Y_est))/length(Y_test);
% MAE_prediction_prune(trial) = mae_prediction2;
% fprintf('MAE prediction error with pruning: %.2f \n', mae_prediction2);

%% cross validation is applied
% check every commented command for at least 10 runs to find the minimum
% loss value
%cvrtree = crossval(rtree,'Holdout',1);
%cvrtree = crossval(rtreepruned,'Holdout',1);
%cvrtree = crossval(rtree,'Holdout',0.005);
%cvrtree = crossval(rtree,'Holdout',0.003);
%cvrtree = crossval(rtree,'Holdout',0.009);
%cvrtree = crossval(rtreepruned,'Holdout',0.003);
%cvrtree = crossval(rtreepruned,'Holdout',0.005);
%cvrtree = crossval(rtreepruned,'Holdout',0.009);
%kfloss = kfoldLoss(cvrtree);
end

%% Results for no pruning
% plots
figure(1)
subplot(311)
plot(training_error_NoPrune,'g.-','LineWidth',1.5) % mean square error
xlabel('no of trial')
ylabel('error magnitude')
title('training error vs trial number [without Pruning]')
grid on
set(gca,'FontSize',12)
subplot(312)
plot(prediction_error_NoPrune,'b.-','LineWidth',1.5) % root mean square error
xlabel('no of trial')
ylabel('error magnitude')
title('rms prediction error vs trial number [without Pruning]')
grid on
set(gca,'FontSize',12)
subplot(313)
plot(MAE_prediction_NoPrune,'r.-','LineWidth',1.5)
xlabel('no of trial')
ylabel('error magnitude')
title('mean absolute prediction error vs trial number [without Pruning]')
grid on
set(gca,'FontSize',12)
% values
TimeWithoutPruning_max = max(time1)
TimeWithoutPruning_avg = mean(time1)
training_error_NoPrune_max = max(training_error_NoPrune)
prediction_error_NoPrune_max = max(prediction_error_NoPrune)
MAE_prediction_NoPrune_max = max(MAE_prediction_NoPrune)
training_error_NoPrune_avg = mean(training_error_NoPrune)
prediction_error_NoPrune_avg = mean(prediction_error_NoPrune)
MAE_prediction_NoPrune_avg = mean(MAE_prediction_NoPrune)
% 
% %% Results for pruning
% % plots
% figure(2)
% subplot(311)
% plot(training_error_prune,'g.-','LineWidth',1.5) % mean square error
% xlabel('no of trial')
% ylabel('error magnitude')
% title('training error vs trial number [with Pruning]')
% grid on
% set(gca,'FontSize',12)
% subplot(312)
% plot(prediction_error_prune,'b.-','LineWidth',1.5) % root mean square error
% xlabel('no of trial')
% ylabel('error magnitude')
% title('rms prediction error vs trial number [with Pruning]')
% grid on
% set(gca,'FontSize',12)
% subplot(313)
% plot(MAE_prediction_prune,'r.-','LineWidth',1.5)
% xlabel('no of trial')
% ylabel('error magnitude')
% title('mean absolute prediction error vs trial number [with Pruning]')
% grid on
% set(gca,'FontSize',12)
% % values
% TimeWithPruning_max = max(time2)
% TimeWithPruning_avg = mean(time2)
% training_error_prune_max = max(training_error_prune)
% prediction_error_prune_max = max(prediction_error_prune)
% MAE_prediction_prune_max = max(MAE_prediction_prune)
% training_error_prune_avg = mean(training_error_prune)
% prediction_error_prune_avg = mean(prediction_error_prune)
% MAE_prediction_prune_avg = mean(MAE_prediction_prune)
% %kfloss





