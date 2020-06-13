%% Color temperature interpolation
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%

function [I_F,I_D,I_C] = colorTempInterpolate(I_T,I_S,ColorTemperatures)
if nargin ==2
    load('ColorTemperatures.mat');
end
cct1 = ColorTemperatures.T;
cct2 = ColorTemperatures.S;
% interpolation weight
cct1inv = 1 / cct1;
cct2inv = 1 / cct2;

tempinv_F = 1 / ColorTemperatures.F;
tempinv_D = 1 / ColorTemperatures.D;
tempinv_C = 1 / ColorTemperatures.C;

g_F = (tempinv_F - cct2inv) / (cct1inv - cct2inv);
g_D = (tempinv_D - cct2inv) / (cct1inv - cct2inv);
g_C = (tempinv_C- cct2inv) / (cct1inv - cct2inv);

I_F = g_F .* I_T + (1-g_F) .* I_S;
I_D = g_D .* I_T + (1-g_D) .* I_S;
I_C = g_C .* I_T + (1-g_C) .* I_S;
end

