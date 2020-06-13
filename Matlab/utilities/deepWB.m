%% Deep white-balance editing main function (inference phase)
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%


function [varargout] = deepWB(I,net,target,device, depth, S, reportTime, normalize)
warning off
if nargin == 2
    target = 'all';
    device = 'gpu';
    depth = 4;
    S = 656;
    reportTime = 0;
    normalize = 0;
elseif nargin == 3
    device = 'gpu';
    depth = 4;
    S = 656;
    reportTime = 0;
    normalize = 0;
elseif nargin == 4
    depth = 4;
    S = 656;
    reportTime = 0;
    normalize = 0;
elseif nargin == 5
    S = 656;
    reportTime = 0;
    normalize = 0;
elseif nargin == 6
    reportTime = 0;
    normalize = 0;
elseif nargin == 7
    normalize = 0;
end

I_resized = imresize(I,[round(size(I,1)/max(size(I))*S)...
    round(size(I,2)/max(size(I))*S)]);

sz = size(I_resized);

if mod(sz(1),2^depth) == 0
    new_size_1 = sz(1);
else
    new_size_1 = sz(1) + 2^depth - mod(sz(1),2^depth);
end

if mod(sz(2),2^depth) == 0
    new_size_2 = sz(2);
else
    new_size_2 = sz(2) + 2^depth - mod(sz(2),2^depth);
end

inSz = [new_size_1, new_size_2, 3];

if sum(sz == inSz) ~= 3
    I_resized = imresize(I_resized,[inSz(1) inSz(2)]);
end

if isfield(net,'AWB_net') == 0
    [AWB_net,T_net,S_net]  = splitNetworks(net, depth);
else
    AWB_net = net.AWB_net;
    T_net = net.T_net;
    S_net = net.S_net;
end

switch lower(target)
    case 'all'
        if reportTime == 1
            tic;
        end
        
        if strcmpi(device,'gpu') == 1  && canUseGPU
            I_resized = gpuArray(dlarray(I_resized,'SSC'));
            I_AWB = gather(extractdata(predict(AWB_net,I_resized)));
            I_T = gather(extractdata(predict(T_net,I_resized)));
            I_S = gather(extractdata(predict(S_net,I_resized)));
        else
            I_resized = dlarray(I_resized,'SSC');
            I_AWB = extractdata(predict(AWB_net,I_resized));
            I_T = extractdata(predict(T_net,I_resized));
            I_S = extractdata(predict(S_net,I_resized));
        end
        
        
        I_resized = imresize(I,[300,300]);
        I_AWB_resized = imresize(I_AWB,[300,300]);
        I_T_resized = imresize(I_T,[300,300]);
        I_S_resized = imresize(I_S,[300,300]);
        
        m_AWB = phi(reshape(I_resized,[],3))\...
            reshape(I_AWB_resized,[],3);
        I_AWB = reshape(phi(reshape(I,[],3)) * m_AWB,size(I));
        I_AWB = out_of_gamut_clipping(I_AWB);
        
        m_T = phi(reshape(I_resized,[],3))\...
            reshape(I_T_resized,[],3);
        I_T = reshape(phi(reshape(I,[],3)) * m_T,size(I));
        I_T = out_of_gamut_clipping(I_T);
        
        m_S = phi(reshape(I_resized,[],3))\...
            reshape(I_S_resized,[],3);
        I_S = reshape(phi(reshape(I,[],3)) * m_S,size(I));
        I_S = out_of_gamut_clipping(I_S);
        
        if normalize == 0
            varargout{1} = I_AWB; varargout{2} = I_T; varargout{3} = I_S;
        else
            I_AWB = I_AWB./(sqrt(sum(I_AWB.^2,3))) .* (sqrt(sum(I.^2,3)));
            varargout{1} = I_AWB;
            I_T = I_T./(sqrt(sum(I_T.^2,3))) .* (sqrt(sum(I.^2,3)));
            varargout{2} = I_T;
            I_S = I_S./(sqrt(sum(I_S.^2,3))) .* (sqrt(sum(I.^2,3)));
            varargout{3} = I_S;
        end
        if reportTime == 1
            toc
        end
        
    case 'awb'
        
        if reportTime == 1
            tic
        end
        if strcmpi(device,'gpu') == 1  && canUseGPU
            I_resized = gpuArray(dlarray(I_resized,'SSC'));
            I_AWB = gather(extractdata(predict(AWB_net,I_resized)));
        else
            I_resized = dlarray(I_resized,'SSC');
            I_AWB = extractdata(predict(AWB_net,I_resized));
        end
        
        if sum(sz==inSz) ~= 3
            I_AWB = imresize(I_AWB,[sz(1) sz(2)]);
        end
        
        I_resized = imresize(I,[300,300]);
        I_AWB_resized = imresize(I_AWB,[300,300]);
        
        m_AWB = phi(reshape(I_resized,[],3))\...
            reshape(I_AWB_resized,[],3);
        I_AWB = reshape(phi(reshape(I,[],3)) * m_AWB,size(I));
        I_AWB = out_of_gamut_clipping(I_AWB);
        
        
        if normalize == 0
            varargout{1} = I_AWB;
        else
            I_AWB = I_AWB./(sqrt(sum(I_AWB.^2,3))) .* (sqrt(sum(I.^2,3)));
            varargout{1} = I_AWB;
        end
        if reportTime == 1
            toc
        end
    case 'editing'
        if reportTime == 1
            tic
        end
        if strcmpi(device,'gpu') == 1  && canUseGPU
            I_resized = gpuArray(dlarray(I_resized,'SSC'));
            I_T = gather(extractdata(predict(T_net,I_resized)));
            I_S = gather(extractdata(predict(S_net,I_resized)));
        else
            I_resized = dlarray(I_resized,'SSC');
            I_T = extractdata(predict(T_net,I_resized));
            I_S = extractdata(predict(S_net,I_resized));
        end
        
        I_resized = imresize(I,[300,300]);
        I_T_resized = imresize(I_T,[300,300]);
        I_S_resized = imresize(I_S,[300,300]);
        
        
        m_T = phi(reshape(I_resized,[],3))\...
            reshape(I_T_resized,[],3);
        I_T = reshape(phi(reshape(I,[],3)) * m_T,size(I));
        I_T = out_of_gamut_clipping(I_T);
        
        m_S = phi(reshape(I_resized,[],3))\...
            reshape(I_S_resized,[],3);
        I_S = reshape(phi(reshape(I,[],3)) * m_S,size(I));
        I_S = out_of_gamut_clipping(I_S);
        
        if normalize == 0
            varargout{1} = I_T; varargout{2} = I_S;
        else
            
            I_T = I_T./(sqrt(sum(I_T.^2,3))) .* (sqrt(sum(I.^2,3)));
            varargout{1} = I_T;
            I_S = I_S./(sqrt(sum(I_S.^2,3))) .* (sqrt(sum(I.^2,3)));
            varargout{2} = I_S;
        end
        if reportTime == 1
            toc
        end
end

