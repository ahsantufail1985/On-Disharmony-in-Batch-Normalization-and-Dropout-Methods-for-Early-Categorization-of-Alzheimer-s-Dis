% DISHARMONY BETWEEN BATCH NORMALIZATION AND DROPOUT CASE-3
% ADD a DROPOUT LAYER AFTER EVERY BN LAYER. AFTER ADDING a DROPOUT LAYER,
% ADD a CONVOLUTIONAL LAYER.
clc
clear all
close all
filters = 8;
layers = [...
    image3dInputLayer([79 95 69],"Name","image3dinput",'Normalization','zerocenter')
    convolution3dLayer([3 3 3],filters,"Name","conv3d_1","Padding","same",  'WeightL2Factor', 0.00005, 'BiasL2Factor', 0.00005)
    batchNormalizationLayer("Name","batchnorm_1")
    eluLayer(1,"Name","elu_1")
    maxPooling3dLayer([2 2 2],"Name","maxpool3d_1","Padding","same", 'Stride',2)
    
    dropoutLayer(0.1,"Name","dropout1")
    convolution3dLayer([3 3 3],filters,"Name","conv3d_2","Padding","same",'WeightL2Factor', 0.00005, 'BiasL2Factor', 0.00005)
    batchNormalizationLayer("Name","batchnorm_2")
    eluLayer(1,"Name","elu_2")
    maxPooling3dLayer([2 2 2],"Name","maxpool3d_2","Padding","same", 'Stride',2)
    
    dropoutLayer(0.1,"Name","dropout2")
    convolution3dLayer([3 3 3],filters,"Name","conv3d_3","Padding","same",'WeightL2Factor', 0.00005, 'BiasL2Factor', 0.00005)
    batchNormalizationLayer("Name","batchnorm_3")
    eluLayer(1,"Name","elu_3")
    maxPooling3dLayer([2 2 2],"Name","maxpool3d_3","Padding","same", 'Stride',2)
    
    dropoutLayer(0.1,"Name","dropout3")
    convolution3dLayer([3 3 3],filters,"Name","conv3d_4","Padding","same",'WeightL2Factor', 0.00005, 'BiasL2Factor', 0.00005)
    batchNormalizationLayer("Name","batchnorm_4")
    eluLayer(1,"Name","elu_4")
    maxPooling3dLayer([2 2 2],"Name","maxpool3d_4","Padding","same", 'Stride',2)
    
    dropoutLayer(0.1,"Name","dropout4")
    convolution3dLayer([3 3 3],filters,"Name","conv3d_5","Padding","same",'WeightL2Factor', 0.00005, 'BiasL2Factor', 0.00005)
    batchNormalizationLayer("Name","batchnorm_5")
    eluLayer(1,"Name","elu_5")
    maxPooling3dLayer([2 2 2],"Name","maxpool3d_5","Padding","same", 'Stride',2)
    
    
    fullyConnectedLayer(300,"Name","fc_1",'WeightL2Factor', 0.00005, 'BiasL2Factor', 0.00005)
    fullyConnectedLayer(100,"Name","fc_2",'WeightL2Factor', 0.00005, 'BiasL2Factor', 0.00005)
    fullyConnectedLayer(3,"Name","fc_3",'WeightL2Factor', 0.00005, 'BiasL2Factor', 0.00005)
    %dropoutLayer(0.1,"Name","dropout")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

TRAINING_SET = imageDatastore('D:\ahsan\TOBECOPIED\temp\FOLD-1\TRAIN', 'IncludeSubfolders',true, 'LabelSource', 'foldernames', 'FileExtensions','.mat','ReadFcn',@(x) matRead(x));
VALIDATION_SET = imageDatastore('D:\ahsan\TOBECOPIED\temp\FOLD-1\VAL', 'IncludeSubfolders',true, 'LabelSource', 'foldernames', 'FileExtensions','.mat','ReadFcn',@(x) matRead(x));
TRAINING_SET = shuffle(TRAINING_SET);
miniBatchSize = 2;
valFrequency = floor(numel(TRAINING_SET.Files)/miniBatchSize);
options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',50, ...
    'Shuffle','every-epoch', ...
    'ValidationData',VALIDATION_SET, ...
    'LearnRateDropPeriod',10,...
    'ValidationFrequency',valFrequency, ...
    'ResetInputNormalization',false, ...
    'MiniBatchSize',miniBatchSize, ...
    'LearnRateSchedule','piecewise',...
    'ExecutionEnvironment','gpu',...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(TRAINING_SET,layers,options);
[Validation_set_predictions,scores] = classify(net,VALIDATION_SET);
