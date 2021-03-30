%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% code to extract spherical harnomic coefficient for 3D tumor
%%%% segmentations
%%%% Written by Jia Wu, MD Anderson Cancer Center
%%%% last updated 3/2021
%%%% contact jwu11@mdanderson.org for any questions

clc; clear all; close all

% setup  the working directory
currentDir = pwd;
dataDir = fullfile(currentDir, 'Processed_data');
binaryDir = fullfile(currentDir, 'SPHARM_output\seg_orig');
fixDir = fullfile(currentDir, 'SPHARM_output\seg_fix');
CodeDir = fullfile(currentDir, 'third_party_package\SPHARM-MAT-v1-0-0\code\');
paraDir = fullfile(currentDir, 'SPHARM_output\seg_para\');
spharmDir = fullfile(currentDir, 'SPHARM_output\seg_spharm');
foeDir = fullfile(currentDir, 'SPHARM_output\seg_foe');
saveDir = fullfile(currentDir, 'SPHARM_output\results');
funDir = fullfile(currentDir, 'Function');
cDir = fullfile(currentDir, 'third_party_package\SPHARM-MAT-v1-0-0\code\C_sources\');
addpath(CodeDir);addpath(funDir);addpath(cDir);

segEvDir = 'C:\SoftwareNoInstall\SegEvaluation\';
contour2Dir = 'D:\Research\Data\Lung_deeplearning\2D_segmetation_QZhang\';

% set them all to 1 when run new cases
processmat_label = 1;
topologyfix_label = 1;
spharmpara_label = 1;
spharm_label = 1;
align_label = 1;
coeff_label = 1;

folder = {'Decathlon'};
%% STEP 1: convert the nifti to mat
% loop through all the autosegmentation files
if processmat_label == 1
    for n_fd = 1:length(folder)
        fileDir = fullfile(dataDir,folder{n_fd});
        cd(fileDir)
        file_name = [folder{n_fd}, '_data.csv'];
        Table = readtable(file_name );
        matDir = fullfile(fileDir,'data');
        
        for i = 1:2  %size(Table,1), process only 2 patients
            % load the nifti files
            if Table.run(i)
                cd(matDir)
                tmp = Table.matFile(i);
                pt = tmp{1};
                disp(pt);
                load(pt); % load the segmentation
                v = tumor1;
                vxsize = img_resolution;
                if sum(v(:)) == 0
                    continue
                end
                s = regionprops3(v,'BoundingBox','Centroid');
                dim = s.BoundingBox;
                v_dim = size(v);
                origin = floor([dim(2) dim(1) dim(3)]);
                bim = v(max(1,origin(1)):min((origin(1)+dim(5)+2),v_dim(1)),max(1,origin(2)):min((origin(2)+dim(4)+2),v_dim(2)),max(1,origin(3)):min((origin(3)+dim(6)+2),v_dim(3))); % crop into the segmentation
                [bim, vxsize] = checkBIM(bim,origin,vxsize);
                cd(binaryDir)
                new_name = [folder{n_fd},'_',extractBefore(pt,'.mat'), '_bim.mat'];
                save(new_name, 'bim', 'origin', 'vxsize');
            end
        end
    end
end

%% STEP 2: Perform the in-house topology fix
if topologyfix_label == 1
    % (1) load the file into a cell
    inFiles = dir(fullfile(binaryDir, '*_bim.mat')); inNames={};
    for i=1:length(inFiles)
        inNames{end+1} = fullfile(binaryDir,inFiles(i).name);
    end
    
    % (2) Display input objects (optional)
    %Available values for Space- 'object';'param';'both'
    dispConfs.Space = 'object';
    % Available values for Mesh- 'orig';'quad32';'quad64';'quad128';'quad256';'quad512';'icosa1';'icosa2'; ...
    %    'icosa3';'icosa4';'icosa5';'icosa6'
    dispConfs.Mesh = 'orig';
    % Available values for Shape- 'solid';'mesh';'both'
    dispConfs.Shade = 'both';
    % Available values for Overlay- 'none';'adc_paramap'
    dispConfs.Overlay = 'none';
    % Available values for Export- 'screen';'png';'both'
    dispConfs.Export = 'png';
    dispConfs.Degree = [];
    dispConfs.Template = '';
    
    SpharmMatUtilDisplayObjs(dispConfs, inNames, CodeDir);
    
    % (3) Perform in-house topology fix
    % Avaliable values for Connectivity - '(6+,18)';'(18,6+)';'(6,26)';'(26,6)'
    confs.Connectivity='(6+,18)';
    confs.Epsilon = 1.5000;
    confs.OutDirectory = fixDir;
    if ~exist(confs.OutDirectory,'dir')
        mkdir(confs.OutDirectory);
    end
    outNames = SpharmMatUtilTopologyFix(confs, inNames, 'InHouse_Fix');
    
    % (4) Display output objects (optional)
    %Available values for Space- 'object';'param';'both'
    dispConfs.Space = 'object';
    % Available values for Mesh- 'orig';'quad32';'quad64';'quad128';'quad256';'quad512';'icosa1';'icosa2'; ...
    %    'icosa3';'icosa4';'icosa5';'icosa6'
    dispConfs.Mesh = 'orig';
    % Available values for Shape- 'solid';'mesh';'both'
    dispConfs.Shade = 'both';
    % Available values for Overlay- 'none';'adc_paramap'
    dispConfs.Overlay = 'none';
    % Available values for Export- 'screen';'png';'both'
    dispConfs.Export = 'png';
    dispConfs.Degree = [];
    dispConfs.Template = '';
    
    SpharmMatUtilDisplayObjs(dispConfs, outNames, CodeDir);
    
end

%% STEP 3: spherical parameterization for surface mesh
if spharmpara_label == 1
    % (1) load the file into a cell
    inFiles = dir(fullfile(fixDir, '*_fix.mat')); inNames={};
    for i=1:length(inFiles)
        inNames{end+1} = fullfile(fixDir, inFiles(i).name);
    end
    
    % (2) Perform Voxel Surfaces (CALD)
    confs.MeshGridSize = 50;
    confs.MaxSPHARMDegree = 6;
    confs.Tolerance = 2;
    confs.Smoothing = 2;
    confs.Iteration = 100;
    confs.LocalIteration = 10;
    % Available values for t_major- 'x';'y'
    confs.t_major = 'x';
    % Available values for SelectDiagonal- 'ShortDiag';'LongDiag'
    confs.SelectDiagonal = 'ShortDiag';
    confs.OutDirectory = paraDir;
    
    if ~exist(confs.OutDirectory,'dir')
        mkdir(confs.OutDirectory);
    end
    
    for i = 1:length(inNames)
        [path, name, ext] = fileparts(inNames{i});
        name  = extractBefore(name,"_fix");
        tmp = [paraDir, name, '_CALD_smo.mat'];
        if exist(tmp, 'file') == 2
            disp([name, '   exist!!!!!!!!!!!']);
            continue
        else
            outNames = SpharmMatParameterization(confs, {inNames{i}}, 'ParamCALD');
        end
    end
    
        % (3) Display output objects (optional)
        %Available values for Space- 'object';'param';'both'
        dispConfs.Space = 'both';
        % Available values for Mesh- 'orig';'quad32';'quad64';'quad128';'quad256';'quad512';'icosa1';'icosa2'; ...
        %    'icosa3';'icosa4';'icosa5';'icosa6'
        dispConfs.Mesh = 'orig';
        % Available values for Shape- 'solid';'mesh';'both'
        dispConfs.Shade = 'both';
        % Available values for Overlay- 'none';'adc_paramap'
        dispConfs.Overlay = 'adc_paramap';
        % Available values for Export- 'screen';'png';'both'
        dispConfs.Export = 'png';
        dispConfs.Degree = [];
        dispConfs.Template = '';
    
        SpharmMatUtilDisplayObjs(dispConfs, outNames, CodeDir);
    
        inFiles = dir([confs.OutDirectory '/initParamCALD/*.mat']); inNames={};
        for i=1:length(inFiles)
            inNames{end+1} = [confs.OutDirectory '/initParamCALD/' inFiles(i).name];
        end
        SpharmMatUtilDisplayObjs(dispConfs, inNames, CodeDir);
end


%% decompose the shape with SPHARM
if spharm_label == 1
    % (1) load the file into a cell
    inFiles = dir(fullfile(paraDir, '*_smo.mat')); inNames={};
    for i=1:length(inFiles)
        inNames{end+1} = fullfile(paraDir, inFiles(i).name);
    end
    
        % (2) Display input objects (optional)
        %Available values for Space- 'object';'param';'both'
        dispConfs.Space = 'object';
        % Available values for Mesh- 'orig';'quad32';'quad64';'quad128';'quad256';'quad512';'icosa1';'icosa2'; ...
        %    'icosa3';'icosa4';'icosa5';'icosa6'
        dispConfs.Mesh = 'orig';
        % Available values for Shape- 'solid';'mesh';'both'
        dispConfs.Shade = 'both';
        % Available values for Overlay- 'none';'adc_paramap'
        dispConfs.Overlay = 'none';
        % Available values for Export- 'screen';'png';'both'
        dispConfs.Export = 'png';
        dispConfs.Degree = [];
        dispConfs.Template = '';
    
        SpharmMatUtilDisplayObjs(dispConfs, inNames, CodeDir);
    
        % (3) Perform SPHARM-MAT Expansion
        confs.MaxSPHARMDegree = 15;
        confs.OutDirectory = spharmDir;
        if ~exist(confs.OutDirectory,'dir')
            mkdir(confs.OutDirectory);
        end
    
        outNames = SpharmMatExpansion(confs, inNames, 'ExpLSF');
    
    % (4) Display output objects (optional)
    %Available values for Space- 'object';'param';'both'
    dispConfs.Space = 'object';
    % Available values for Mesh- 'orig';'quad32';'quad64';'quad128';'quad256';'quad512';'icosa1';'icosa2'; ...
    %    'icosa3';'icosa4';'icosa5';'icosa6'
    dispConfs.Mesh = 'icosa4';
    % Available values for Shape- 'solid';'mesh';'both'
    dispConfs.Shade = 'both';
    % Available values for Overlay- 'none';'adc_paramap'
    dispConfs.Overlay = 'none';
    % Available values for Export- 'screen';'png';'both'
    dispConfs.Export = 'png';
    dispConfs.Degree = 15;
    dispConfs.Template = '';
    
    outNames={};
    for i = 1:length(inNames)
        [path, name, ext] = fileparts(inNames{i});
        name  = extractBefore(name,"_smo");
        outNames{end+1} = strcat(spharmDir, '/', name, '_LSF_des.mat');
    end
    
    SpharmMatUtilDisplayObjs(dispConfs, outNames, CodeDir);
    
end

%% align the surface with FOE
if align_label == 1
    % (1) load the file into a cell
    inFiles = dir(fullfile(spharmDir, '*_des.mat')); inNames={};
    for i=1:length(inFiles)
        inNames{end+1} = fullfile(spharmDir, inFiles(i).name);
    end
    
    % (4) Perform FOE Alignment
    % Available values for CPoint- 'x';'y';'z'
    confs.CPoint = 'y';
    % Available values for NPole- 'x';'y';'z'
    confs.NPole = 'z';
    confs.MaxSPHARMDegree = 15;
    confs.OutDirectory = foeDir;
    
    if ~exist(confs.OutDirectory,'dir')
        mkdir(confs.OutDirectory);
    end
    
    outNames = SpharmMatAlignment(confs, inNames, 'AligFOE');
    
        % (5) Display output objects (optional)
        %Available values for Space- 'object';'param';'both'
        dispConfs.Space = 'object';
        % Available values for Mesh- 'orig';'quad32';'quad64';'quad128';'quad256';'quad512';'icosa1';'icosa2'; ...
        %    'icosa3';'icosa4';'icosa5';'icosa6'
        dispConfs.Mesh = 'quad64';
        % Available values for Shape- 'solid';'mesh';'both'
        dispConfs.Shade = 'both';
        % Available values for Overlay- 'none';'adc_paramap'
        dispConfs.Overlay = 'adc_paramap';
        % Available values for Export- 'screen';'png';'both'
        dispConfs.Export = 'png';
        dispConfs.Degree = 1;
        dispConfs.Template = '';
    
        SpharmMatUtilDisplayObjs(dispConfs, outNames, CodeDir);
    
    
        dispConfs.Degree = 12;
        SpharmMatUtilDisplayObjs(dispConfs, outNames, CodeDir);
    
end


%% align the surface with FOE
if coeff_label == 1
    % (1) load the file into a cell
    inFiles = dir(fullfile(foeDir, '*_reg.mat')); inNames={};
    for i=1:length(inFiles)
        inNames{end+1} = fullfile(foeDir, inFiles(i).name);
    end
    
    % extract the spherical transformed coefficient
    [name, radius, radius_norm, shapeScale, shapeScale_norm, energy_dgr, energy_dgr_norm, energy_skewness, tumor_vol, spharm_coeff] = spharm_coeff(inNames, fixDir, 15);
    
    new_name = fullfile(saveDir, folder{1});
    save(new_name, 'name', 'radius', 'radius_norm', 'shapeScale', 'shapeScale_norm', 'energy_dgr', 'energy_dgr_norm', 'energy_skewness', 'tumor_vol', 'spharm_coeff');
    
end
