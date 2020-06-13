%% output net layer
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%


classdef outLayer < nnet.layer.Layer
    
    properties
    end
    
    properties (Learnable)
    end
    
    methods
        function layer = outLayer(name)
            layer.Name = name;
            %bypass layer
            layer.Description = "outLayer";
            
        end
        
        function X = predict(layer, X)
            %Z = X;
        end
        
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            dLdX = dLdZ;
            clear X dLdZ Z
        end
        
    end
end