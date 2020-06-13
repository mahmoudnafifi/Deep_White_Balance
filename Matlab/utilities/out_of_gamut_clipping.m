%% out of gamut mapping
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%


function [I,map] = out_of_gamut_clipping(I)

    sz = size(I);
    
    I = reshape(I,[],3);

     map = ones(size(I,1),1);
     map(I(:,1)>1 | I(:,2)>1 | I(:,3)>1 | I(:,1)<0 | I(:,2)<0 | I(:,3)<0)=0;
     map = reshape(map,[sz(1),sz(2)]);
    
     I(I(:,1)>1)=1; 
     I(I(:,1)<0)=0; 

     I = reshape(I,[sz(1),sz(2),sz(3)]);

end

