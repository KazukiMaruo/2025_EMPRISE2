function BpRF_run_pRF_analysis(subj_id, session, mod_name, ana_name, avg, conf, voxels)
% _
% Estimate Voxel-Wise Population Receptive Field Model
% FORMAT BpRF_run_pRF_analysis(subj_id, session, mod_name, voxels)
% 
%     subj_id  - a string, subject ID (e.g. "EDY7")
%     session  - a string, session name (e.g. "visual")
%     mod_name - a string, model name (e.g. "base")
%     ana_name - a string, analysis name (e.g. "
%     avg      - a logical indicating whether to average over runs
%     conf     - an n x c x r array of confound variables
%     voxels   - a  1 x v vector of voxel indices to estimate
% 
% FORMAT BpRF_run_pRF_analysis(subj_id, session, mod_name, voxels)
% 
% Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
% 2023-07-06, 09:34: first version
% 2023-07-06, 19:01: introduced voxels
% 2023-09-11, 15:45: introduced analysis name, confounds, averaging


%%% Step 0: set global parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load project directories
dirs = load('project_directories.mat');

% specify data IDs
sub  = subj_id;
ses  = session;
task = 'harvey';
acq1 = 'mprageised';
acq2 = 'fMRI1p75TE24TR2100iPAT3FS';
desc = 'preproc';
run  = [1:8];
model= mod_name;
ana  = ana_name;                % pRF analysis name

% specify fMRI parameters
TR  = 2.1;                      % repetition time
TE  = 0.024;                    % echo time
mtr = 41;                       % microtime resolution
mto = 21;                       % microtime onset

% get derivatives directory
deri_dir = strcat(dirs.data_dir,'derivatives/');

% get statistics directory
glm_dir  = strcat(deri_dir,'spm12/','sub-',sub,'/','ses-',ses,'/','model-',model,'/');
num_runs = numel(run);


%%% Step 1: prepare model inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load onsets and durations
filename = strcat(glm_dir,'sub-',sub,'_','ses-',ses,'_','model-',model,'_','run-',num2str(run(1)),'_','onsets.mat');
stims    = load(filename);

% sort stimuli by onset time
stim_mat = [];
for i = 1:numel(stims.names)
    num = i; if i == numel(stims.names), num = 20; end;
    stim_mat = [stim_mat; [stims.onsets{i}, stims.durations{i}, num*ones(size(stims.onsets{i}))]];
end;
stim_mat = sortrows(stim_mat,1);
clear num

% prepare stimulus time series
for j = 1:size(stim_mat,1)
    % numerosity stimulus fields
    U(j).num   = stim_mat(j,3);
    U(j).ons   = stim_mat(j,1);
    U(j).dur   = stim_mat(j,2);
    U(j).dt    = TR/mtr;
    % BayespRF "legacy fields"
    U(j).dist  = U(j).num;
    U(j).angle = 1;
    U(j).pmax  = 5.5;
    U(j).pmin  = 0.5;
end;
    
% Alternatively, you may wish to customise the stimulus presentation details (i.e. create your own version of prepare_inputs_polar_samsrf.m). This script needs to produce a Matlab structure array with the following fields:
% 
%     U(t).dist  = dist;         % Distance
%     U(t).angle = angle;        % Angle
%     U(t).ons = TR * (t-1);     % Onset (secs)
%     U(t).dur = TR;             % Duration (secs)
%     U(t).dt  = TR/nmicrotime;  % 1 bin=1/16 second, i.e. 16 bins per second
%     U(t).pmax = stim_diameter; % Stimulus diameter
%     U(t).pmin = 0.5;           % Minimum PRF size
% 
% The structure U will have one entry per time point (t) of the stimulus. Time points are indexed from 1. For example, if a bar crosses the display in 20 steps, and this happens 100 times, there will be 2000 entries in U.
% 
% The U structure contains fields which describe the stimulus at time t:
% 
%     Dist and angle are vectors of the polar coordinates illuminated at this time step.
%     Ons and dur specify the onset and duration of this stimulus in seconds. Here we've set this to assume one frame of the stimulus per TR (the time taken to acquire an MRI volume), but this need not be the case.
%     dt is the length of each microtime bin in seconds (typically nmicrotime=16). Modelling is conducted at a finer timescale than MRI acquisitions. Each stimulus time step will be divided into short 'microtime' bins of length dt seconds.
%     pmax is the stimulus diameter in degrees of visual angle
%     pmin is the minimum entertained pRF size in degrees of visual angle

% load volumes of interest
xY = cell(num_runs,1);
for i = 1:num_runs
    xY{i} = strcat(glm_dir,'VOI_model-',model,'_run-',num2str(run(i)),'_timeseries.mat');
end


%%% Step 2: specify pRF model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load SPM.mat
load(strcat(glm_dir,'SPM.mat'));
SPM.swd = glm_dir;

% get confounds
if nargin < 6 || isempty(conf)
    X0 = ones(SPM.nscan(1),1,num_runs);
else
    X0 = conf;
end;

% specify options
options = struct('TE',         TE, ...
                 'voxel_wise', true, ...
                 'name',       ana_name, ...
                 'model',      strcat('spm_prf_fcn_','numerosity'), ...
                 'avg_sess',   avg, ...
                 'avg_method','mean', ...
                 'X0',         X0, ...
                 'B0',         7); % 7T
% options = struct(...
%      'TE',TE, ...           % echo time
%      'voxel_wise',false, ...% per voxel (true) or ROI (false)
%      'model',model,...      % pRF function (spm_prf_fcn...)
%      'hE',6, ...            % expected log precision of noise
%      'P',[], ...            % starting parameters
%      'B0',3, ...            % fMRI field strength (teslas)
%      'avg_sess',true, ...   % average sessions' timeseries
%      'avg_method','mean',...% accepts 'mean' or 'eigen'
%      'X0',X0, ...           % matrix of nuisance regressors
%      'delays', TR / 2, ...  % microtime offset
%      'pE', struct, ...      % (optional) prior means
%      'pC', struct, ...      % (optional) prior variance
%      'name', char);         % (optional) name of PRF file

% specify pRF model
PRF = spm_prf_analyse('specify', SPM, xY, U, options);

% correct stimulus bins
load(strcat(glm_dir,'PRF_',ana,'.mat'));
for j = 1:numel(PRF.U)
    PRF.U(j).ind = round(PRF.U(j).ind);
end;
save(strcat(glm_dir,'PRF_',ana,'.mat'), 'PRF');


%%% Step 3: estimate pRF model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% option 1: for selected voxels
if nargin > 7 && ~isempty(voxels)

    % estimate pRF model
    options  = struct('voxels', voxels);
    filename = strcat(glm_dir,'PRF_',ana,'.mat');
    PRF_est  = spm_prf_analyse('estimate', filename, options);
    
    % review pRF model
    spm_prf_review(filename, voxels(1));
    
end;

% option 2: for all voxels
if nargin < 7 || isempty(voxels)
    
    % estimate pRF model
    options  = struct('nograph', true, 'use_parfor', true);
    filename = strcat(glm_dir,'PRF_',ana,'.mat');
    PRF_est  = spm_prf_analyse('estimate', filename, options);

    % review pRF model
    spm_prf_review(filename);
    
end;

% go back to tools
cd(dirs.tool_dir);


%%% Step 4: do some postprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% import labels to NIfTI
% hemi = 'lh';
% reg  = 'V1';
% surf_dir   = strcat(deri_dir,'freesurfer/','sub-',sub,'_ses-','audio','/surf/');
% label_file = strcat(surf_dir,hemi,'_',reg,'.label');
% spm_prf_import_label(label_file, glm_dir);

% plot summed response in ROI
% figure('Color', 'w');
% prf_file = strcat(glm_dir,'PRF_',ana,'.mat');
% roi_file = strcat(glm_dir,hemi,'_',reg,'.nii');
% load(prf_file);
% spm_prf_summarise(PRF, roi_file);
% title('Region of interest','FontSize', 16);

% compute negative entropy map
% prf_file = strcat(glm_dir,'PRF_',ana,'.mat');
% load(prf_file);
% spm_prf_plot_entropy(PRF, {'dist','angle'}, 'dist_angle', true);