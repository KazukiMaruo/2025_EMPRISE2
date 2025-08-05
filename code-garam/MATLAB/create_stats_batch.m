function create_stats_batch(subj_id, session, mod_name, run_batch)
% _
% Create SPM batch editor job for first-level fMRI analysis
% FORMAT create_stats_batch(subj_id, session, mod_name)
% 
%     subj_id   - a string, subject ID (e.g. "EDY7")
%     session   - a string, session name (e.g. "visual")
%     mod_name  - a string, name of the GLM (e.g. "base")
%     run_batch - a logical, indicating whether to run the batch
% 
% FORMAT create_stats_batch(subj_id, session, mod_name) creates a job for
% first-level fMRI analysis of selected subject, session and model and
% saves them nto an SPM batch editor file. If run_batch is true, then this
% analysis is performed.
% 
% Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
% 2023-05-25, 19:03: first version
% 2023-05-25, 20:37: changed derivatives directory
% 2023-06-08, 13:50: introduced model name, implemented reference slice
% 2023-06-15, 16:08: implemented contrasts for model "para"
% 2023-07-04, 14:24: implemented contrasts for model "pRF"


%%% Step 1: specify settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set running, if necessary
if nargin < 4 || isempty(run_batch)
    run_batch = true;
end;

% load project directories
dirs = load('project_directories.mat');

% specify numbers
num_runs = 8;
num_scan = 145;

% specify IDs
sub  = subj_id;
ses  = session;
task = 'harvey';
acq  = 'fMRI1p75TE24TR2100iPAT3FS';
run  = [1:num_runs];
space= 'T1w';
desc = 'preproc';
model= mod_name;

% specify fMRI
n    = num_scan;                % number of scans
TR   = 2.1;                     % repetition time
HPF  = 128;                     % high-pass filter cut-off

% calculate reference slice
slice_timing = [0, 1.075, 0.05, 1.125, 0.1025, 1.1775, 0.1525, 1.2275, 0.2025, 1.28, 0.255, 1.33, 0.305, 1.3825, 0.3575, 1.4325, 0.4075, 1.485, 0.46, 1.535, 0.51, 1.5875, 0.5625, 1.6375, 0.6125, 1.69, 0.665, 1.74, 0.715, 1.79, 0.7675, 1.8425, 0.8175, 1.8925, 0.87, 1.945, 0.92, 1.995, 0.9725, 2.0475, 1.0225];
start_time   =  1.024;
temp_diff    = abs(sort(slice_timing)-start_time);
rSl          = find(temp_diff==min(temp_diff));
nSl          = numel(slice_timing);


%%% Step 2: create batch %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% prepare directories
deri_dir  = strcat(dirs.data_dir,'derivatives/spm12/');
if ~exist(deri_dir,'dir'), mkdir(deri_dir); end;
subj_dir = strcat(deri_dir,'sub-',sub,'/');
if ~exist(subj_dir,'dir'), mkdir(subj_dir); end;
sess_dir = strcat(subj_dir,'ses-',ses,'/');
if ~exist(sess_dir,'dir'), mkdir(sess_dir); end;
mod_dir  = strcat(sess_dir,'model-',model,'/');
if ~exist(mod_dir,'dir'), mkdir(mod_dir); end;

% specify GLM directory
matlabbatch{1}.spm.stats.fmri_spec.dir = {mod_dir};

% specify timing parameters
matlabbatch{1}.spm.stats.fmri_spec.timing.units   = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT      = TR;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t  = nSl;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = rSl;

% specify fMRI runs
for i = 1:num_runs
    
    % specify fMRI scans
    bold_nii = cell(num_scan,1);
    for j = 1:num_scan
        bold_nii{j} = strcat(dirs.prep_dir,'sub-',sub,'/','ses-',ses,'/func/',...
                             'sub-',sub,'_','ses-',ses,'_','task-',task,'_','acq-',acq,'_',...
                             'run-',num2str(run(i)),'_','space-',space,'_','desc-',desc,'_','bold.nii',',',num2str(j));
    end;
    matlabbatch{1}.spm.stats.fmri_spec.sess(i).scans = bold_nii;
    clear bold_nii
    
    % specify multiple conditions
    onsets_mat = strcat(mod_dir,'sub-',sub,'_','ses-',ses,'_','model-',model,'_','run-',num2str(run(i)),'_','onsets.mat');
    matlabbatch{1}.spm.stats.fmri_spec.sess(i).cond  = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(i).multi = {onsets_mat};
    if i == 1, onsets_mat_1 = onsets_mat; end;
    clear onsets_mat
    
    % specify multiple regressors
    regressors_mat = strcat(mod_dir,'sub-',sub,'_','ses-',ses,'_','model-',model,'_','run-',num2str(run(i)),'_','regressors.mat');
    matlabbatch{1}.spm.stats.fmri_spec.sess(i).regress   = struct('name', {}, 'val', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(i).multi_reg = {regressors_mat};
    clear regressors_mat
    
    % specify high-pass filter
    matlabbatch{1}.spm.stats.fmri_spec.sess(i).hpf = HPF;
    
end;

% specify all the rest
matlabbatch{1}.spm.stats.fmri_spec.fact             = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt             = 1;
matlabbatch{1}.spm.stats.fmri_spec.global           = 'None';
matlabbatch{1}.spm.stats.fmri_spec.mthresh          = 0.8;
matlabbatch{1}.spm.stats.fmri_spec.mask             = {''};
matlabbatch{1}.spm.stats.fmri_spec.cvi              = 'AR(1)';

% specify model estimation
matlabbatch{2}.spm.stats.fmri_est.spmmat(1)        = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals  = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

% specify contrast manager
matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));

% specify EOI contrasts (1)
onsets_1 = load(onsets_mat_1);
num_cons = numel(onsets_1.names);
if strcmp(model,'para'), num_cons = num_cons + 1; end;
clear onsets_1 onsets_mat_1

% specify EOI contrasts (2)
matlabbatch{3}.spm.stats.con.consess{1}.fcon.name    = 'EOI';
matlabbatch{3}.spm.stats.con.consess{1}.fcon.weights = eye(num_cons);
matlabbatch{3}.spm.stats.con.consess{1}.fcon.sessrep = 'repl';
matlabbatch{3}.spm.stats.con.consess{2}.fcon.name    = 'sum<>base';
matlabbatch{3}.spm.stats.con.consess{2}.fcon.weights = ones(1,num_cons);
matlabbatch{3}.spm.stats.con.consess{2}.fcon.sessrep = 'repl';

% specify other contrasts (1)
if strncmp(model,'base',4)
    con_types =  'FttFtFtFtFtFt';
    con_names = {'few<>many', 'few>many', 'many>few', ...
                 'one<>other', 'one>other', 'two<>other', 'two>other', ...
                 'three<>other', 'three>other', 'four<>other', 'four>other', ...
                 'five<>other', 'five>other'};
    con_vecs  = [ 1  1  1  1  1 -5;
                  1  1  1  1  1 -5;
                 -1 -1 -1 -1 -1  5;
                  4 -1 -1 -1 -1  0;
                  4 -1 -1 -1 -1  0;
                 -1  4 -1 -1 -1  0;
                 -1  4 -1 -1 -1  0;
                 -1 -1  4 -1 -1  0;
                 -1 -1  4 -1 -1  0;
                 -1 -1 -1  4 -1  0;
                 -1 -1 -1  4 -1  0;
                 -1 -1 -1 -1  4  0;
                 -1 -1 -1 -1  4  0];
elseif strcmp(model,'para')
    con_types =  'FttFtt';
    con_names = {'few<>many', 'few>many', 'many>few', ...
                 'numerosity', 'num-pos', 'num-neg'};
    con_vecs  = [ 1  0 -1;
                  1  0 -1;
                 -1  0  1;
                  0  1  0;
                  0  1  0;
                  0 -1  0];
elseif strcmp(model,'pRF')
    con_types = '';
    con_names = {};
    con_vecs  = [];
end;

% specify other contrasts (2)
for k = 1:numel(con_names)
    if con_types(k) == 'F';
        matlabbatch{3}.spm.stats.con.consess{2+k}.fcon.name    = con_names{k};
        matlabbatch{3}.spm.stats.con.consess{2+k}.fcon.weights = con_vecs(k,:);
        matlabbatch{3}.spm.stats.con.consess{2+k}.fcon.sessrep = 'repl';
    elseif con_types(k) == 't';
        matlabbatch{3}.spm.stats.con.consess{2+k}.tcon.name    = con_names{k};
        matlabbatch{3}.spm.stats.con.consess{2+k}.tcon.weights = con_vecs(k,:);
        matlabbatch{3}.spm.stats.con.consess{2+k}.tcon.sessrep = 'repl';
    end;
end;

% finalize contrast manager
matlabbatch{3}.spm.stats.con.delete = 0;


%%% Step 3: save batch %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% save batch editor job
filename = strcat(mod_dir,'sub-',sub,'_','ses-',ses,'_','model-',model,'_','design.mat');
save(filename, 'matlabbatch');

% run batch editor job
if run_batch
    fprintf('\n\n-> Running stats batch "%s":', filename);
    spm_jobman('run', filename);
end;