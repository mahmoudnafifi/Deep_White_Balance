%% preparing training data
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%


function [Trdata,Vldata] = getTrVlData(fold,datasetDir, patchesPerImage, ...
    patchSize, aug, trainImgNum, valRatio)

if nargin == 2
    patchesPerImage = 4;
    patchSize = [128 128];
    aug = 0;
    trainImgNum = 12000;
    valRatio = 0.1;
elseif nargin == 3
    patchSize = [128 128];
    aug = 0;
    trainImgNum = 12000;
    valRatio = 0.1;
elseif nargin == 4
    aug = 0;
    trainImgNum = 12000;
    valRatio = 0.1;
elseif nargin == 5
    trainImgNum = 12000;
    valRatio = 0.1;
elseif nargin == 6
    valRatio = 0.1;
end

if fold ~= 0
load(fullfile('..','folds',sprintf('fold%d.mat',fold)));
fileNames = {};
counter= 1;
for i = 1 : length(data.training)
    tempFiles = dir(fullfile(datasetDir,data.training{i}));
    for j = 1 : length(tempFiles)
        fileNames{counter} = fullfile(datasetDir,tempFiles(j).name);
        counter = counter + 1;
    end
end

else
    fileNames = dir(fullfile(datasetDir,'*.png'));
    fileNames = fullfile(datasetDir,{fileNames(:).name});
end

inds = randperm(length(fileNames));
fileNames = fileNames (inds);

tr_fileNames = fileNames(1: end - floor(length(fileNames)*valRatio));


if trainImgNum ~= 0 && trainImgNum < length(tr_fileNames)
   tr_fileNames = tr_fileNames(1:trainImgNum);
end


trainingIn = imageDatastore(tr_fileNames, 'ReadFcn',@inRead);

trainingGT = imageDatastore(tr_fileNames, 'ReadFcn',@gtRead);

vl_fileNames = fileNames(end - floor(length(fileNames)*valRatio) + 1 : end);

validationIn = imageDatastore(vl_fileNames, 'ReadFcn',@inRead);
validationGT = imageDatastore(vl_fileNames, 'ReadFcn',@gtRead);


if aug == 1
    augmenter = imageDataAugmenter( ...
        'RandRotation',@()randi([0,1],1)*90, ...
        'RandXReflection',true);
    
    Trdata = randomPatchExtractionDatastore(trainingIn,trainingGT, ...
        patchSize,'DataAugmentation',augmenter,'PatchesPerImage',...
        patchesPerImage);
elseif aug == 0
    Trdata = randomPatchExtractionDatastore(trainingIn,trainingGT, ...
        patchSize,'PatchesPerImage', patchesPerImage);
end

Vldata = randomPatchExtractionDatastore(validationIn,validationGT, ...
        patchSize,'PatchesPerImage', patchesPerImage);

