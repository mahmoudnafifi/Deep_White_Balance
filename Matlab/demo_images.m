%% Demo processing image directory
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%

close all;
clc
clear;

load('ColorTemperatures.mat'); %color temperatures

basedir = fullfile('..','example_images'); %input directory

in_ext = '.jpg';

outdir = fullfile('..','result_images'); %output directory

model_filename = fullfile('models','nets.mat'); %model name

depth = 4; %depth of encoder/decoder

device = 'gpu'; %'cpu' or 'gpu'

S = 656; %max dim of input image to the network, the output will be saved in its original resolution

target = 'AWB'; %task -- use 'All', 'AWB', or 'Editing'

target_color_temp = []; % select a specific color temperature [2850 - 7500]. Remember to make the task variable = 'Editing'.

normalize = 0; %post-processing to normalize each pixel intensity according to input pixel intensity

report_time = 0; %1 to report processing time

show = 0; %1 to visualize each image -- note: it pauses the program after visualizing each image


if exist(outdir,'dir') == 0
    
    mkdir(outdir);
    
end

if isempty(target_color_temp) == 0
    if target_color_temp > 7500 || target_color_temp < 2850
        error('Color temperature should be in the range [2850 - 7500], but the given one is %d!',target_color_temp);
    end
    
    if strcmpi(target, 'editing') == 0
        error('The task should be editing when a target color temperature is specified.');
    end
end

images = dir(fullfile(basedir,['*' in_ext]));

fprintf('Loading...\n');

load(model_filename);

if isfield(net,'AWB_net') == 0
    
    [AWB_net,T_net,S_net]  = splitNetworks(net, depth);
    
    clear net;
    
    net.AWB_net = AWB_net;
    
    net.T_net = T_net;
    
    net.S_net = S_net;
end

for i = 1 : length(images)
    
    in_img_filename = fullfile(basedir,images(i).name);
    
    [~,name,ext] = fileparts(in_img_filename);
    
    fprintf('Reading image: %s...\n',in_img_filename);
    
    I = im2double(imread(in_img_filename));
    
    if length(size(I))<3
        
        error('Input image is grayscale');
        
    elseif size(I,3) ~=3
        
        error('Input image should have three RGB color channels');
        
    end
    
    fprintf('Processing image: %s...\n',in_img_filename);
    
    switch lower(target)
        
        case 'all'
            
            [I_AWB,I_T,I_S] = deepWB(I,net,target,device,depth, S, ...
                report_time, normalize);
            
            [I_F,I_D,I_C] = colorTempInterpolate(I_T,I_S,ColorTemperatures);
            
            if show == 1
                
                fprintf('Visualizing results for image: %s press any key to continue...\n',in_img_filename);
                
                figure('units','normalized','outerposition',[0 0 1 1]);
                
                subplot(3,3,1);imshow(I); title('input');
                
                subplot(3,3,2);imshow(I_AWB); title('white-balanced');
                
                subplot(3,3,3);imshow(I_T); title('Tungsten WB');
                
                subplot(3,3,4);imshow(I_F); title('Fluorescent WB');
                
                subplot(3,3,5);imshow(I_D); title('Daylight WB');
                
                subplot(3,3,6);imshow(I_C); title('Cloudy WB');
                
                subplot(3,3,8);imshow(I_S); title('Shade WB');
                
                pause;
                
                close all;
                
            end
            
            imwrite(I_AWB,fullfile(outdir,[name '_AWB.png']));
            
            imwrite(I_T,fullfile(outdir,[name '_T.png']));
            
            imwrite(I_S,fullfile(outdir,[name '_S.png']));
            
            imwrite(I_F,fullfile(outdir,[name '_F.png']));
            
            imwrite(I_C,fullfile(outdir,[name '_C.png']));
            
            imwrite(I_D,fullfile(outdir,[name '_D.png']));
            
        case 'awb'
            
            I_AWB = deepWB(I,net,target,device,depth, S, report_time, ...
                normalize);
            
            if show == 1
                
                fprintf('Visualizing result for image: %s press any key to continue...\n',in_img_filename);
                
                figure('units','normalized','outerposition',[0 0 1 1]);
                
                subplot(1,2,1);imshow(I); title('input');
                
                subplot(1,2,2);imshow(I_AWB); title('white-balanced');
                
                pause;
                
                close all;
                
            end
            
            imwrite(I_AWB,fullfile(outdir,[name '_AWB.png']));
            
        case 'editing'
            
            [I_T,I_S] = deepWB(I,net,target,device,depth, S, ...
                report_time, normalize);
            
            if isempty(target_color_temp)
                [I_F,I_D,I_C] = colorTempInterpolate(I_T,I_S,ColorTemperatures);
                
                if show == 1
                    
                    fprintf('Visualizing results for image: %s press any key to continue...\n',in_img_filename);
                    
                    figure('units','normalized','outerposition',[0 0 1 1]);
                    
                    subplot(2,3,1);imshow(I); title('input');
                    
                    subplot(2,3,2);imshow(I_T); title('Tungsten WB');
                    
                    subplot(2,3,3);imshow(I_F); title('Fluorescent WB');
                    
                    subplot(2,3,4);imshow(I_D); title('Daylight WB');
                    
                    subplot(2,3,5);imshow(I_C); title('Cloudy WB');
                    
                    subplot(2,3,6);imshow(I_S); title('Shade WB');
                    
                    pause;
                    
                    close all;
                    
                end
                
                imwrite(I_T,fullfile(outdir,[name '_T.png']));
                
                imwrite(I_S,fullfile(outdir,[name '_S.png']));
                
                imwrite(I_F,fullfile(outdir,[name '_F.png']));
                
                imwrite(I_C,fullfile(outdir,[name '_C.png']));
                
                imwrite(I_D,fullfile(outdir,[name '_D.png']));
                
            else
                output = colorTempInterpolate_w_target(I_T,I_S,target_color_temp);
                
                if show == 1
                    
                    fprintf('Visualizing result for image: %s press any key to continue...\n',in_img_filename);
                    
                    figure('units','normalized','outerposition',[0 0 1 1]);
                    
                    subplot(1,2,1);imshow(I); title('input');
                    
                    subplot(1,2,2);imshow(output);
                    
                    title(sprintf('output (%dK)', target_color_temp));
                    
                    pause;
                    
                    close all;
                    
                end
                
                if isempty(outdir) == 0
                    
                    imwrite(output,fullfile(outdir,[name ...
                        sprintf('_%d.png',target_color_temp)]));
                    
                end
            end
            
    end
    
    fprintf('Done!\n');
    
end

