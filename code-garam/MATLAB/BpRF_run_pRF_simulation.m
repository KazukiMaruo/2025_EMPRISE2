function BpRF_run_pRF_simulation(subj_id, session, mod_name, ana_name, sim_file, avg, conf, voxels)
% _
% Estimate Voxel-Wise Population Receptive Field Model
% FORMAT BpRF_run_pRF_simulation(subj_id, session, mod_name, ana_name, sim_file, avg, conf, voxels)
% 
%     subj_id  - a string, subject ID (e.g. "EDY7")
%     session  - a string, session name (e.g. "visual")
%     mod_name - a string, model name (e.g. "base")
%     ana_name - a string, simulation identifier (e.g. "Simulation_A")
%     sim_file - a string, absolute path to simulated signals
%     avg      - a logical indicating whether to average over runs
%     conf     - a logical indicating whether to include covariates
%     voxels   - a vector of voxel indices to estimate
% 
% FORMAT BpRF_run_pRF_simulation(subj_id, session, mod_name, ana_name, sim_file, avg, conf, voxels)
% 
% Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
% 2023-07-10, 09:32: first version
% 2023-08-17, 13:45: second version; added analysis name, confound variables
% 2023-09-07, 18:40: third version; added confound selection, averaging choice


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
model= mod_name;                % pRF analysis name
ana  = strcat(ana_name,'_',num2str(avg),'_',num2str(conf));

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

% load simulation file
Sim = load(sim_file);

% create volumes of interest
v    = size(Sim.Y,2);
VOIs = cell(num_runs,1);
for i = 1:num_runs
    VOIs{i}.Y  = mean(Sim.Y(:,:,i),2);
    VOIs{i}.xY = struct();
    VOIs{i}.xY.name  = sprintf('simulation_run-%d', i);
    VOIs{i}.xY.Ic    = 1;
    VOIs{i}.xY.Sess  = i;
    VOIs{i}.xY.XYZmm = zeros(3,v);
    VOIs{i}.xY.X0    = ones(size(Sim.Y,1),1);
    VOIs{i}.xY.y     = Sim.Y(:,:,i);
    VOIs{i}.xY.v     = 1/v*ones(v,1);
    % other fields of "xY":
    %       xyz: [3×1 double]
    %       def: 'mask'
    %      spec: [1×1 struct]
    %       str: 'image mask: .\\VOI\_model-pRF\_run\_mask.nii'
    %         u: [145×1 double]
    %         s: [145×1 double]
end;


%%% Step 2: specify pRF model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load SPM.mat
load(strcat(glm_dir,'SPM.mat'));
SPM.swd  = glm_dir;

% get confounds
if conf && isfield(Sim,'X')
    X0 = Sim.X(:,2:end,:);
else
    X0 = ones(size(Sim.Y,1),1,num_runs);
end;

% specify options
options = struct('TE',         TE, ...
                 'voxel_wise', true, ...
                 'name',       ana, ...
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
PRF = spm_prf_analyse_JS('specify', SPM, VOIs, U, options);

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
    PRF_est  = spm_prf_analyse_JS('estimate', filename, options);
    
    % review pRF model
    spm_prf_review(filename, voxels(1));
    
end;

% option 2: for all voxels
if nargin < 8 || isempty(voxels)
    
    % estimate pRF model
    options  = struct('nograph', true, 'use_parfor', true);
    filename = strcat(glm_dir,'PRF_',ana,'.mat');
    PRF_est  = spm_prf_analyse_JS('estimate', filename, options);

    % review pRF model
    % spm_prf_review(filename);
    
end;

% go back to tools
cd(dirs.tool_dir);