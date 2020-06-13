%% Uninstall
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%

disp('Uninstalling...')
current = pwd;
rmpath(fullfile(current,'arch'));
rmpath(fullfile(current,'utilities'));
savepath
disp('Done!');
