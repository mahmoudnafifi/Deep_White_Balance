%% Replace input layer in the network
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%

function net = replaceInputLayer(net,inputsize)
net = replaceLayer(net,'InputLayer',imageInputLayer(inputsize,'Name',...
    'InputLayer','Normalization','none'));
end

