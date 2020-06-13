%% Color temperature interpolation
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%
function output = colorTempInterpolate_w_target(I_T,I_S,target_temp)

cct1 = 2850;
cct2 = 7500;
% interpolation weight
cct1inv = 1 / cct1;
cct2inv = 1 / cct2;

tempinv_target = 1 / target_temp;


g = (tempinv_target - cct2inv) / (cct1inv - cct2inv);


output = g .* I_T + (1-g) .* I_S;

end

