function save_(result, fileName)
% Custom save function to use inside parfor loops
%
%Input:
%   -data: data to be saved.
%   -fileName: full fileName
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
save(fileName,'result');
end

