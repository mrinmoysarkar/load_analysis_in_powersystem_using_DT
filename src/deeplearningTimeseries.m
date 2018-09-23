%% author: mrinmoy sarkar
% email: mrinmoy.pol@gmail.com
%%
clear all;
close all;
load trainv2.mat

y_observed = zeros(9,15);
y_predict = zeros(9,15);
for location=1:9
    fprintf('Location: %d\n',location)
    data = trainDataset(trainDataset(:,1)==location,4);
    data = data';
    % figure
    % plot(data)
    % xlabel("time")
    % ylabel("load")
    % title("load vs time")
    
    numTimeStepsTrain = floor(0.9*numel(data));% 90 percent for training
    
    dataTrain = data(1:numTimeStepsTrain+1);
    dataTest = data(numTimeStepsTrain+1:end);
    
    mu = mean(dataTrain);
    sig = std(dataTrain);
    
    dataTrainStandardized = (dataTrain - mu) / sig;
    
    XTrain = dataTrainStandardized(1:end-1);
    YTrain = dataTrainStandardized(2:end);
    
    
    numFeatures = 1;
    numResponses = 1;
    numHiddenUnits = 200;
    
    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits)
        fullyConnectedLayer(numResponses)
        regressionLayer];
    
    options = trainingOptions('adam', ...
        'MaxEpochs',250, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',0, ...
        'Plots','training-progress');
    
    net = trainNetwork(XTrain,YTrain,layers,options);
    
    dataTestStandardized = (dataTest - mu) / sig;
    XTest = dataTestStandardized(1:end-1);
    
    net = predictAndUpdateState(net,XTrain);
    [net,YPred] = predictAndUpdateState(net,YTrain(end));
    
    numTimeStepsTest = numel(XTest);
    for i = 2:numTimeStepsTest
        [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
    end
    
    YPred = sig*YPred + mu;
    
    YTest = dataTest(2:end);
    rmse = sqrt(mean((YPred-YTest).^2))
    
    figure
    plot(dataTrain(1:end-1))
    hold on
    idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
    plot(idx,[data(numTimeStepsTrain) YPred],'.-')
    hold off
    xlabel("time")
    ylabel("load")
    title("Forecast")
    legend(["Observed" "Forecast"])
    
    figure
    subplot(2,1,1)
    plot(YTest)
    hold on
    plot(YPred,'.-')
    hold off
    legend(["Observed" "Forecast"])
    ylabel("load")
    title("Forecast")
    
    subplot(2,1,2)
    stem(YPred - YTest)
    xlabel("time")
    ylabel("Error")
    title("RMSE = " + rmse)
    
    
    net = resetState(net);
    net = predictAndUpdateState(net,XTrain);
    
    YPred = [];
    numTimeStepsTest = numel(XTest);
    for i = 1:numTimeStepsTest
        [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
    end
    
    YPred = sig*YPred + mu;
    
    rmse = sqrt(mean((YPred-YTest).^2))
    
    figure
    subplot(2,1,1)
    plot(YTest)
    hold on
    plot(YPred,'.-')
    hold off
    legend(["Observed" "Predicted"])
    ylabel("load")
    title("Forecast with Updates")
    
    subplot(2,1,2)
    stem(YPred - YTest)
    xlabel("time")
    ylabel("Error")
    title("RMSE = " + rmse)
    
    y_observed(location,:) = YTest;
    y_predict(location,:) = YPred;
    
    figure(100)
    subplot(3,3,location)
    plot(y_observed(location,:))
    hold on
    plot(y_predict(location,:),'.-')
    hold off
    legend(["Observed" "Predicted"])
    title(strcat('location ', num2str(location)))
    grid on
end