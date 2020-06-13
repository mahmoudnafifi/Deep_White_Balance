%% GUI demo
% Author: Mahmoud Afifi
% Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
%%

function varargout = demo_GUI(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @demo_GUI_OpeningFcn, ...
    'gui_OutputFcn',  @demo_GUI_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end




function demo_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
global nets
handles.output = hObject;
guidata(hObject, handles);
handles.status.String = 'Loading...';pause(0.001);
handles.browse_btn.Enable = 'On';
handles.wb_btn.Enable = 'Off';
handles.save_btn.Enable = 'Off';
handles.tint_slider.Enable = 'Off';
handles.shade_slider.Enable = 'Off';
handles.temp_slider.Enable = 'Off';
handles.normalization.Enable = 'Off';
handles.custom_rbtn.Enable = 'Off';
handles.auto_rbtn.Enable = 'Off';
nets = load(fullfile('models','nets.mat'));
handles.status.String = 'Ready!';pause(0.001);
handles.status.String = '';pause(0.001);
handles.output = hObject;
guidata(hObject, handles);


function varargout = demo_GUI_OutputFcn(hObject, eventdata, handles)

varargout{1} = handles.output;



function browse_btn_Callback(hObject, eventdata, handles)
global I File_Name Path_Name I_T I_S I_AWB
[File_Name, Path_Name] = uigetfile({'*.jpg';'*.png';'*.jpeg'},...
    'Select input image',fullfile('..','example_images'));
if File_Name ~=0
    I = im2double(imread(fullfile(Path_Name,File_Name)));
    handles.status.String = 'Loading image...';pause(0.001);
    axes(handles.image);
    imshow(I);
    I_T = [];
    I_S = [];
    I_AWB = [];
    handles.wb_btn.Enable = 'On';
    handles.save_btn.Enable = 'On';
    handles.tint_slider.Enable = 'Off';
    handles.shade_slider.Enable = 'Off';
    handles.temp_slider.Enable = 'On';
    handles.normalization.Enable = 'On';
    handles.custom_rbtn.Enable = 'On';
    handles.auto_rbtn.Enable = 'On';
    handles.status.String = 'Done!';pause(0.001);
    handles.status.String = '';pause(0.001);
end

function wb_btn_Callback(hObject, eventdata, handles)
global I I_AWB I_T I_S output nets
handles.status.String = 'Processing...';pause(0.001);
[I_AWB,I_T,I_S] = deepWB(I,nets.net, 'all', 'gpu', 4, 656, 0, 0);
error = 0;
if handles.auto_rbtn.Value == 1
    output = I_AWB;
elseif handles.custom_rbtn.Value == 1
    target_temp = 2850 + (7500 - 2850) * handles.temp_slider.Value;
    output = colorTempInterpolate_w_target(I_T,I_S,target_temp);
else
    handles.status.String = 'Error!';
    error = 1;
    pause(1);
    handles.status.String = '';
end
if error == 0
    axes(handles.image);
    imshow(output);
end
handles.tint_slider.Enable = 'On';
handles.shade_slider.Enable = 'On';
handles.status.String = 'Done!';pause(0.001); handles.status.String = '';pause(0.001);


function save_btn_Callback(hObject, eventdata, handles)
global File_Name output Path_Name
[~,name,ext] = fileparts(File_Name);
outFile_Name = [name '_wb_edited' ext];
[file,path,~] = uiputfile({'*.jpg';'*.png';'*.jpeg';'*.*'},...
    'Save Image',fullfile(Path_Name,outFile_Name));
if file ~=0
    handles.browse_btn.Enable = 'Off';
    handles.wb_btn.Enable = 'Off';
    handles.save_btn.Enable = 'Off';
    handles.tint_slider.Enable = 'Off';
    handles.shade_slider.Enable = 'Off';
    handles.temp_slider.Enable = 'Off';
    handles.normalization.Enable = 'Off';
    handles.custom_rbtn.Enable = 'Off';
    handles.auto_rbtn.Enable = 'Off';
    handles.status.String = 'Processing...';pause(0.001);
    imwrite(output, fullfile(path,file));
    handles.status.String = 'Done!';
    pause(0.01); handles.status.String = '';
    handles.browse_btn.Enable = 'On';
    handles.wb_btn.Enable = 'On';
    handles.save_btn.Enable = 'On';
    handles.tint_slider.Enable = 'On';
    handles.shade_slider.Enable = 'On';
    handles.temp_slider.Enable = 'On';
    handles.normalization.Enable = 'On';
    handles.custom_rbtn.Enable = 'On';
    handles.auto_rbtn.Enable = 'On';
end


function tint_slider_Callback(hObject, eventdata, handles)
global output I_T I_S I_AWB
handles.status.String = 'Processing...';
error = 0;
if handles.auto_rbtn.Value == 1
    output = I_AWB;
elseif handles.custom_rbtn.Value == 1
    target_temp = 2850 + (7500 - 2850) * handles.temp_slider.Value;
    output = colorTempInterpolate_w_target(I_T,I_S,target_temp);
else
    handles.status.String = 'Error!';
    error = 1;
    pause(1);
    handles.status.String = '';
end
if error== 0
    ill = mean(reshape(output,[],3));
    ind = find(ill == max(ill));
        output(:,:,ind) = output(:,:,ind) + (1 - ill(ind)) * ...
            handles.tint_slider.Value;
    axes(handles.image);
    imshow(output);
    handles.status.String = 'Done!';  pause(0.1); handles.status.String = '';
end

function shade_slider_Callback(hObject, eventdata, handles)
global output I_AWB I_T I_S
handles.status.String = 'Processing...';
error = 0;
if handles.auto_rbtn.Value == 1
    output = I_AWB;
elseif handles.custom_rbtn.Value == 1
    target_temp = 2850 + (7500 - 2850) * handles.temp_slider.Value;
    output = colorTempInterpolate_w_target(I_T,I_S,target_temp);
else
    handles.status.String = 'Error!';
    error = 1;
    pause(1);
    handles.status.String = '';
end
if error== 0
    output(:,:,3) = output(:,:,3) .* (1 - handles.shade_slider.Value);
    axes(handles.image);
    imshow(output);
    handles.status.String = 'Done!';  pause(0.1); handles.status.String = '';
end



function normalization_Callback(hObject, eventdata, handles)
global I I_AWB I_T I_S output
handles.status.String = 'Processing...';
error = 0;
if handles.auto_rbtn.Value == 1
    output = I_AWB;
elseif handles.custom_rbtn.Value == 1
    target_temp = 2850 + (7500 - 2850) * handles.temp_slider.Value;
    output = colorTempInterpolate_w_target(I_T,I_S,target_temp);
else
    handles.status.String = 'Error!';
    error = 1;
    pause(1);
    handles.status.String = '';
end
if error == 0
    
    if handles.normalization == 1
        output = output./(sqrt(sum(output.^2,3))) .* (sqrt(sum(I.^2,3)));
    end
end
handles.status.String = 'Done!';  pause(0.1); handles.status.String = '';


function temp_slider_Callback(hObject, eventdata, handles)
global I_T I_S output
handles.status.String = 'Processing...';
handles.custom_rbtn.Value = 1;
target_temp = 2850 + (7500 - 2850) * handles.temp_slider.Value;
output = colorTempInterpolate_w_target(I_T,I_S,target_temp);
axes(handles.image);
imshow(output);
handles.status.String = 'Done!';  pause(0.1); handles.status.String = '';

function auto_rbtn_Callback(hObject, eventdata, handles)
global I_AWB output
if isempty(I_AWB) == 0
    handles.status.String = 'Processing...';
    output = I_AWB;
    axes(handles.image);
    imshow(output);
    handles.status.String = 'Done!';  pause(0.1); handles.status.String = '';
end

function custom_rbtn_Callback(hObject, eventdata, handles)
global I_T I_S output
if isempty(I_T) == 0 || isempty(I_S) == 0
    handles.status.String = 'Processing...';
    target_temp = 2850 + (7500 - 2850) * handles.temp_slider.Value;
    output = colorTempInterpolate_w_target(I_T,I_S,target_temp);
    axes(handles.image);
    imshow(output);
    handles.status.String = 'Done!';  pause(0.1); handles.status.String = '';
end

%%%
function temp_slider_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function tint_slider_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function shade_slider_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
