%% Splitting trained single-encoder-multi-decoder into three networks 
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%

function [AWB_net,T_net,S_net]  = splitNetworks(net, depth)

net = layerGraph(net);

AWB_net = net;
T_net = net;
S_net = net;

clear net;

prefixs= {'T-','S-'};
for p = 1 : length(prefixs)
    AWB_net = removeLayers(AWB_net,{AWB_net.Layers(...
        contains({AWB_net.Layers(:).Name},prefixs(p))).Name});
end

cutindx = find(strcmp({AWB_net.Layers(:).Name},...
    sprintf('Encoder-Stage-%d-MaxPool',depth)));

AWBDec_layer_names = {AWB_net.Layers(cutindx+2:end-1).Name};

T_net = removeLayers(T_net,AWBDec_layer_names);
S_net = removeLayers(S_net,AWBDec_layer_names);

for p = 1 : length(prefixs)
    modelName = prefixs{p};
    modelName = [modelName(1) '_net'];
    curr_prefixs = setdiff(prefixs,prefixs{p});
    for cp = 1 : length(curr_prefixs)
        
        eval(sprintf('%s=removeLayers(%s,{%s.Layers(contains({%s.Layers(:).Name},curr_prefixs{%d})).Name});',...
            modelName,modelName,modelName,modelName,cp));
        
    end
end
T_net = removeLayers(T_net,'FinalLayer');
S_net = removeLayers(S_net,'FinalLayer');
AWB_net = removeLayers( AWB_net,'catLayer');
AWB_net = removeLayers( AWB_net,'outLayer');
AWB_net = removeLayers( AWB_net,'FinalLayer');


T_net = dlnetwork(T_net);
S_net = dlnetwork(S_net);
AWB_net = dlnetwork(AWB_net);