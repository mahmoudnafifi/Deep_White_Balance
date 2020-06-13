%% L1 loss
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%

classdef maeLossLayer < nnet.layer.RegressionLayer
    properties
        scale
    end
    methods
        function layer = maeLossLayer(name)
            layer.Name = name;
            layer.Description = 'Mean absolute error';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Calculate MAE.
            
            sz = size(Y);
            if length(sz) == 3
                R = 1;
            else
                R = sz(4);
            end
            loss = abs(Y-T);
           
            loss = sum(loss(:))/R;
        end
    end
end