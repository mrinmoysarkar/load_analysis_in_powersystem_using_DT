%% author: mrinmoy sarkar
% email: mrinmoy.pol@gmail.com
%%

clear all;
close all;



listing = dir('..\data\');
fileNames = listing(3:8);
path = listing.folder;

trainDataset=zeros(468,4);
indx = 1;
for i=1:length(fileNames)
    filename = strcat(path,'\',fileNames(i).name);
    data = xlsread(filename);
    traindata = data(6,1:26);
    for j=1:length(traindata)
        trainDataset(indx,1) = 1; % 1 means dhaka
        trainDataset(indx,2) = i; % day
        trainDataset(indx,3) = j; % time
        trainDataset(indx,4) = traindata(j); % load
        indx = indx+1;
    end
    traindata = data(7,1:26);
    for j=1:length(traindata)
        trainDataset(indx,1) = 2; % 2 means Chittagong
        trainDataset(indx,2) = i; % day
        trainDataset(indx,3) = j; % time
        trainDataset(indx,4) = traindata(j); % load
        indx = indx+1;
    end
    traindata = data(9,1:26);
    for j=1:length(traindata)
        trainDataset(indx,1) = 3; % 3 means Rajshahi
        trainDataset(indx,2) = i; % day
        trainDataset(indx,3) = j; % time
        trainDataset(indx,4) = traindata(j); % load
        indx = indx+1;
    end
end
save('train.mat','trainDataset')
load train.mat

noOftrial = 100;
trainning_error = zeros(1,noOftrial);
prediction_error = zeros(1,noOftrial);
for trial=1:noOftrial

shuffletrainData = randperm(size(trainDataset,1));
trainSamples = fix(0.8 * length(shuffletrainData));
X = trainDataset(shuffletrainData(1:trainSamples),1:3);
Y = trainDataset(shuffletrainData(1:trainSamples),end);
X_test = trainDataset(shuffletrainData(trainSamples+1:end),1:3);
Y_test = trainDataset(shuffletrainData(trainSamples+1:end),end);


rtree = fitrtree(X,Y);
% uncomment to view the trained tree 
%view(rtree,'mode','graph');
totaltrainingloss = loss(rtree,X,Y)/length(Y);
trainning_error(trial) = totaltrainingloss;
fprintf("total error in training without prunning: %.2f \n", totaltrainingloss);

Y_est = predict(rtree,X_test);
predicterror = (sum((Y_test-Y_est).^2)/length(Y_test)).^0.5;
prediction_error(trial) = predicterror;
fprintf("RMS prediction error without prunning: %.2f \n", predicterror);

alpha = 0.7;
Level = fix(alpha*max(rtree.PruneList));
rtreepruned = prune(rtree,"Level",Level);
% uncomment to view the trained tree
%view(rtreepruned,'mode','graph');
totaltrainingloss = loss(rtreepruned, X, Y)/length(Y);
fprintf("total error in training with prunning: %.2f \n", totaltrainingloss);

Y_est = predict(rtreepruned, X_test);
predicterror = (sum((Y_test-Y_est).^2)/length(Y_test)).^0.5;
fprintf("RMS prediction error with prunning: %.2f \n", predicterror);

% kfold = 50;
% cvrtree = crossval(rtree,'KFold',50);
% kfloss = kfoldLoss(cvrtree);
end

figure(1)
subplot(211)
plot(trainning_error)
xlabel('no of trial')
ylabel('trainning error')
subplot(212)
plot(prediction_error)
xlabel('no of trial')
ylabel('prediction error')



