%% read ground-truth image
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%


function images = gtRead(fileName)

    [directory,basename,ext] = fileparts (fileName); 
    currname = [basename ext];
    parts = strsplit(currname,'_');
    parts = parts(1:end-2);
    basename = '';
    for p = 1 : length(parts)
        basename = [basename parts{p} '_'];
    end
    sname = [basename 'S_AS.png'];
    tname = [basename 'T_AS.png'];
    gtname = [basename 'G_AS.png'];
    
    I_AWB = im2double(imread(fullfile(directory,gtname)));

    images = zeros(size(I_AWB,1),size(I_AWB,2), size(I_AWB,3) * 3, ...
        'like',I_AWB);
    images(:,:,1:3) = I_AWB;
    images(:,:,4:6) = im2double(imread(fullfile(directory,tname)));
    images(:,:,7:9) = im2double(imread(fullfile(directory,sname)));
    
end
