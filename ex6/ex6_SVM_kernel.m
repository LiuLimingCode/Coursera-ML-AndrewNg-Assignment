clear all
close all

% Load from ex6data2: 
% You will have X, y in your environment
load('ex6data3.mat');

% Plot training data
plotData(X, y);

%% SVM kernel
% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);