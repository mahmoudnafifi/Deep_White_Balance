%% Constructs network architecture 
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%


function net = buildNet(imageSize, depth, chnls, convvfilter)

cutedge = depth * 5 + 2;

%main network (encoder  + AWB decoder)
net = removeLayers(unetLayers(imageSize,3, 'EncoderDepth',depth, ...
    'NumFirstEncoderFilters', chnls, 'FilterSize' , convvfilter),...
    {'Softmax-Layer','Segmentation-Layer'});

%removing dropOut layers
net = removeLayers(net,'Bridge-DropOut');
net = removeLayers(net,sprintf('Encoder-Stage-%d-DropOut',depth));
net = connectLayers(net,'Bridge-ReLU-2','Decoder-Stage-1-UpConv');
net = connectLayers(net,sprintf('Encoder-Stage-%d-ReLU-2',depth),...
    sprintf('Encoder-Stage-%d-MaxPool',depth));

inLayer = imageInputLayer(imageSize,'Name','InputLayer',...
    'Normalization','none');

net = replaceLayer(net,'ImageInputLayer',inLayer);


%tungsten 

lgraphT = unetLayers(imageSize,3, 'EncoderDepth',depth, ...
    'NumFirstEncoderFilters', chnls, 'FilterSize' , convvfilter);

%removing dropOut layers
lgraphT = removeLayers(lgraphT,'Bridge-DropOut');
lgraphT = removeLayers(lgraphT,sprintf('Encoder-Stage-%d-DropOut',depth));
lgraphT = connectLayers(lgraphT,'Bridge-ReLU-2','Decoder-Stage-1-UpConv');
lgraphT = connectLayers(lgraphT,sprintf('Encoder-Stage-%d-ReLU-2',depth),...
    sprintf('Encoder-Stage-%d-MaxPool',depth));

lgraphT = removeLayers(lgraphT,{lgraphT.Layers(1:cutedge).Name});

lgraphT = removeLayers(lgraphT,{'Softmax-Layer','Segmentation-Layer'});

T_decoder = lgraphT.Layers;

for i = 1 : length(T_decoder)
    T_decoder(i).Name = ['T-' T_decoder(i).Name];
end

net = addLayers(net,T_decoder);

for i = 1 : depth
    net = connectLayers(net,sprintf('Encoder-Stage-%d-ReLU-2',i),...
        sprintf('T-Decoder-Stage-%d-DepthConcatenation/in2',depth-i+1));
end

 net = connectLayers(net,'Bridge-Conv-1',T_decoder(1).Name);
    
clear T_decoder;

%shade
lgraphS = unetLayers(imageSize,3, 'EncoderDepth',depth, ...
    'NumFirstEncoderFilters', chnls, 'FilterSize' , convvfilter);

%removing dropOut layers
lgraphS = removeLayers(lgraphS,'Bridge-DropOut');
lgraphS = removeLayers(lgraphS,sprintf('Encoder-Stage-%d-DropOut',depth));
lgraphS = connectLayers(lgraphS,'Bridge-ReLU-2','Decoder-Stage-1-UpConv');
lgraphS = connectLayers(lgraphS,sprintf('Encoder-Stage-%d-ReLU-2',depth),...
    sprintf('Encoder-Stage-%d-MaxPool',depth));

lgraphS = removeLayers(lgraphS,{lgraphS.Layers(1:cutedge).Name});

lgraphS = removeLayers(lgraphS,{'Softmax-Layer','Segmentation-Layer'});

S_decoder = lgraphS.Layers;

for i = 1 : length(S_decoder)
    S_decoder(i).Name = ['S-' S_decoder(i).Name];
end

net = addLayers(net,S_decoder);

for i = 1 : depth
    net = connectLayers(net,sprintf('Encoder-Stage-%d-ReLU-2',i),...
        sprintf('S-Decoder-Stage-%d-DepthConcatenation/in2',depth-i+1));
end

 net = connectLayers(net,'Bridge-Conv-1',S_decoder(1).Name);
    
clear S_decoder;

%concatination layer
catLayer = depthConcatenationLayer(3,'Name','catLayer');


output = outLayer('outLayer');

FLayer = maeLossLayer('FinalLayer');

net = addLayers(net,[catLayer output FLayer]);

net = connectLayers(net,'Final-ConvolutionLayer','catLayer/in1');
net = connectLayers(net,'T-Final-ConvolutionLayer','catLayer/in2');
net = connectLayers(net,'S-Final-ConvolutionLayer','catLayer/in3');
    

end

