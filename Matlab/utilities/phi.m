%% Implementation of different polynomial mapping as described in: M Afifi, et al., When Color Constancy Goes Wrong: Correcting Improperly White-Balanced Images, CVPR 19.
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%

function O=phi(I, degree)
if nargin == 1
    degree = 34;
end
switch degree
    case 34
        O=[I,... %r,g,b
            I.*I,... %r2,g2,b2
            I(:,1).*I(:,2),I(:,1).*I(:,3),I(:,2).*I(:,3),... %rg,rb,gb
            I.*I.*I,... %r3,g3,b3
            I(:,1).*(I(:,2).^2),I(:,1).*(I(:,3).^2),... %r(g2),r(b2)
            I(:,2).*(I(:,3).^2),I(:,2).*(I(:,1).^2),... %g(b2),g(r2)
            I(:,3).*(I(:,2).^2),I(:,3).*(I(:,1).^2),... %b(g2),b(r2)
            I(:,1).*I(:,2).*I(:,3),... %rgb
            I.*I.*I.*I,... %r4,g4,b4
            (I(:,1).^3).*(I(:,2)),(I(:,1).^3).*(I(:,3)),... %(r3)g,(r3)(b)
            (I(:,2).^3).*(I(:,1)),(I(:,2).^3).*(I(:,3)),... %(g3)(r),(g3)(b)
            (I(:,3).^3).*(I(:,1)),(I(:,3).^3).*(I(:,2)),... %(b3)(r),(b3)(g)
            (I(:,1).^2).*(I(:,2).^2),(I(:,2).^2).*(I(:,3).^2),... %(r2)(g2),(g2)(b2)
            (I(:,1).^2).*(I(:,3).^2),... %(r2)(b2)
            (I(:,1).^2).*I(:,2).*I(:,3),... %(r2)gb
            (I(:,2).^2).*I(:,1).*I(:,3),... %(g2)rb
            (I(:,3).^2).*I(:,1).*I(:,2)] ;%(b2)rg 
    case 11
        O=[I,... %r,g,b
            I(:,1).*I(:,2),I(:,1).*I(:,3),I(:,2).*I(:,3),... %rg,rb,gb
            I.*I,... %r2,g2,b2
            I(:,1).*I(:,2).*I(:,3),... %rgb
            ones(size(I,1),1)]; %1 
    case 9
        O=[I,... %r,g,b
            I(:,1).*I(:,2),I(:,1).*I(:,3),I(:,2).*I(:,3),... %rg,rb,gb
            I.*I,... %r2,g2,b2
            ];   
end
end