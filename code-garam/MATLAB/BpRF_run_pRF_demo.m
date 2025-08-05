function BpRF_run_pRF_demo(subj_id, session, mod_name, voxels)
% _
% Estimate Voxel-Wise Population Receptive Field Model
% FORMAT BpRF_run_pRF_demo(subj_id, session, mod_name, voxels)
% 
%     subj_id  - a string, subject ID (e.g. "EDY7")
%     session  - a string, session name (e.g. "visual")
%     mod_name - a string, model name (e.g. "base")
%     voxels   - a vector of voxel indices to estimate
% 
% FORMAT BpRF_run_pRF_demo(subj_id, session, mod_name, voxels)
% 
% Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
% 2023-08-07, 09:53: first version


%%% Step 0: set global parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load project directories
dirs = load('project_directories.mat');

% specify data IDs
sub  = subj_id;
ses  = session;
run  = [1:8];
model=  mod_name;
ana  = 'demo';                  % pRF analysis name

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

% load volumes of interest
xY = cell(num_runs,1);
for i = 1:num_runs
    xY{i} = strcat(glm_dir,'VOI_model-',model,'_run-',num2str(run(i)),'_timeseries.mat');
end


%%% Step 2: specify pRF model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load SPM.mat
load(strcat(glm_dir,'SPM.mat'));
SPM.swd  = glm_dir;

% specify options
options = struct('TE',         TE, ...
                 'voxel_wise', true, ...
                 'name',       ana, ...
                 'model',      strcat('spm_prf_fcn_','numerosity'), ...
                 'B0',         7); % 7T

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
if nargin > 3 && ~isempty(voxels)

    % estimate pRF model
    options  = struct('voxels', voxels);
    filename = strcat(glm_dir,'PRF_',ana,'.mat');
    PRF_est  = spm_prf_analyse('estimate', filename, options);
    
    % review pRF model
    spm_prf_review(filename, voxels(1));
    if numel(voxels) > 1
        spm_prf_review(filename, voxels(end));
    end;
    
end;

% option 2: for all voxels
if nargin < 4 || isempty(voxels)
    
    % estimate pRF model
    options  = struct('use_parfor', true);
    filename = strcat(glm_dir,'PRF_',ana,'.mat');
    PRF_est  = spm_prf_analyse('estimate', filename, options);

    % review pRF model
    spm_prf_review(filename);
    
end;

% go back to tools
cd(dirs.tool_dir);