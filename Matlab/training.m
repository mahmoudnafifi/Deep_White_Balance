%% Training code
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%

clc
clear;

datasetDir = fullfile('..','dataset'); % please, adjust image directory accordingly 

fold = 1; %validation fold: 1,2,3 or 0. Set fold = 0 uses all images without cross validation

if fold ~= 0
    fprintf('Training code for validation fold number %d\n',fold);
else
    fprintf('Training code\n');
end

loadpath = [];

patchsPerImg = 4; %a single random patch from each image

trainingImgsNum = 13333; %if 0, then load all training images

validationRatio = 0.1;

patchSize = [128, 128, 3];

spatialAug = 1;

epochs = 110;

miniBatch = 32;

lR = 10^-4;

encoderDecoderDepth = 4;

chnls = 24;

convvfilter = 3;

checkpoint_dir = sprintf('reports_and_checkpoints_%d',fold);

GPUDevice = 1;

L2Reg = 10^-5;

checkpoint_period = 5;

if fold == 0
    modelName = 'model.mat';
else
    modelName = sprintf('model_%d.mat',fold);
end

fprintf('Prepare training data ...\n');

[Trdata,Vldata] = getTrVlData(fold,datasetDir, patchsPerImg, ...
    patchSize(1:2),spatialAug,trainingImgsNum,validationRatio);

options = get_training_options(epochs,miniBatch,lR,...
    checkpoint_dir,Vldata,GPUDevice, L2Reg, checkpoint_period);


if isempty(loadpath) == 1
    fprintf('Create the model ...\n');
    net = buildNet(patchSize, encoderDecoderDepth, chnls, convvfilter);
    
else 
    fprintf('Loading a checkpoint model from %s ...\n', loadpath);
    load(loadpath);
    net = layerGraph(net);
end

fprintf('Start training ...\n');

net = trainNetwork(Trdata,net,options);

fprintf('Saving the trained model ...\n');

if exist('models','dir') == 0
    mkdir('models');
end
save(fullfile('models',sprintf('net_%d.mat',fold)),'net','-v7.3');

fprintf('Models saved in net_%d.mat!\nDone!\n',fold);

fprintf('Saving each auto-encoder model separately ...\n');

[AWB_net,T_net,S_net]  = splitNetworks(net, encoderDecoderDepth);

clear net;
net.AWB_net = AWB_net;
net.T_net = T_net;
net.S_net = S_net;

save(fullfile('models',sprintf('nets_%d.mat',fold)),'net','-v7.3');

fprintf('Models saved in nets_%d.mat!\nYou can access any autoencoder by loading the model and then use net.X.\nFor example, use net.AWB for autoWB model\nDone!\n',fold);