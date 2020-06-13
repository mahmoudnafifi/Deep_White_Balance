function  startpooling(NUM_WORKERS)
%Start parallel pool using the given number of cores (requires Parallel
%Computing Toolbox).
%
%Input:
%   -NUM_WORKERS: number of cores.
%
% Copyright (c) 2018 Mahmoud Afifi
% Lassonde School of Engineering
% York University
% mafifi@eecs.yorku.ca
%
% Permission is hereby granted, free of charge, to any person obtaining 
% a copy of this software and associated documentation files (the 
% "Software"), to deal in the Software with restriction for its use for 
% research purpose only, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
%
% Please cite the following work if this program is used:
% Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown, "White Balance Correction for sRGB Rendered Images", ECCV 2018.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
po = gcp('nocreate');
if ~isempty(po)
    if po.NumWorkers ~= NUM_WORKERS
        delete(po);
        pc=parcluster('local');
        pc.NumWorkers=NUM_WORKERS;
        po = parpool(NUM_WORKERS);
    end
else
    pc=parcluster('local');
    pc.NumWorkers=NUM_WORKERS;
    po = parpool(NUM_WORKERS);
end
end