%% author: mrinmoy sarkar
% email: mrinmoy.pol@gmail.com
% Modified and Tested by Dhiman Chowdhury
% Power (Load/Demand) Data: Dhaka, Chittagong and Rajshahi
%%
clear all;
close all;
warning('off','all');
% path_delimiter = '\'; %for windows
% path_delimiter = '/'; % for mac or linux
% listing = dir(strcat('..',path_delimiter,'data',path_delimiter,'v2'));
% fileNames = listing(3:8);
% path = listing.folder;
% trainDataset=zeros(6*9*26,4);
% indx = 1;
% for i=1:length(fileNames)
%     filename = strcat(path,path_delimiter,fileNames(i).name);
%     data = xlsread(filename);
%     for j=8:16
%         traindata = data(j,3:28);
%         for k=1:length(traindata)
%             trainDataset(indx,1) = j-7; % 1 means dhaka
%             trainDataset(indx,2) = i; % day
%             trainDataset(indx,3) = k; % time
%             trainDataset(indx,4) = traindata(k); % load
%             indx = indx+1;
%         end
%     end
% end
% save('trainv2.mat','trainDataset')



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
%   Mdl = fitrensemble(X,Y,'NumLearningCycles',100,'OptimizeHyperparameters','all'); %random forest
     Mdl = fitrtree(X,Y,'OptimizeHyperparameters','all'); % regression tree
%     rng default;
%     Mdl = fitrsvm(X,Y,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));%svm
    time1(trial) = toc;
    %fprintf('elapsed time_without pruning: %.6f \n', time1);
    %uncomment to view the trained tree (should be viewed and attached to paper)
    %view(Mdl,'mode','graph'); % for regression tree only
    
    %mean squared error
    totaltrainingloss1 = loss(Mdl,X,Y)/length(Y);
    training_error_NoPrune(trial) = totaltrainingloss1;
    %fprintf('total error in training without pruning: %.2f \n', totaltrainingloss1);
    
    Y_est = predict(Mdl,X_test);
    
    %rms prediction error
    predicterror1 = (sum((Y_test-Y_est).^2)/length(Y_test)).^0.5;
    prediction_error_NoPrune(trial) = predicterror1;
    %fprintf('RMS prediction error without pruning: %.2f \n', predicterror1);
    
    %mean absolute error
    mae_prediction1 = sum(abs(Y_test-Y_est))/length(Y_test);
    MAE_prediction_NoPrune(trial) = mae_prediction1;
    %fprintf('MAE prediction error without prunning: %.2f \n', mae_prediction1);
end

plot(Y_test,'r')
hold on
plot(Y_est,'b')
legend("observed","forecast")
fprintf('Average result for %d trials: \n',noOftrial);
fprintf('total error in training: %.2f \n', mean(training_error_NoPrune));
fprintf('RMS prediction error: %.2f \n', mean(prediction_error_NoPrune));
fprintf('MAE prediction error: %.2f \n', mean(MAE_prediction_NoPrune));

