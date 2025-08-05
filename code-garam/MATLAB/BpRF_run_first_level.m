function BpRF_run_first_level(subj_id, session, mod_name)
% _
% Extract Time Series from First-Level Model
% FORMAT run_first_level(subj_id, session, mod_name)
% 
%     subj_id  - a string, subject ID (e.g. "EDY7")
%     session  - a string, session name (e.g. "visual")
%     mod_name - a string, model name (e.g. "base")
% 
% FORMAT run_first_level(subj_id, session, mod_name) (i) imports a
% FreeSurfer cortical surface for subject subj_id; (ii) loads the first-
% level GLM mod_name from subj_id and session and thresholds an effects
% of interest (EOI) contrast to identify voxels responsive to stimulation
% as such; and (iii) extracts time series from these voxels and saves them
% into a volume of interest (VOI) file.
% 
% Note: For the time being, step (i) is ommitted.
% 
% Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
% 2023-07-04, 15:21: first version
% 2023-07-06, 17:50: renamed VOI files


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
model=  mod_name;
hemi = {'lh', 'rh'};

% get derivatives directory
deri_dir = strcat(dirs.data_dir,'derivatives/');

% get statistics directory
glm_dir  = strcat(deri_dir,'spm12/','sub-',sub,'/','ses-',ses,'/','model-',model,'/');

% get anatomical image
strc_img = strcat(deri_dir,'fmriprep/','sub-',sub,'/','ses-','audio','/anat/',...
                  'sub-',sub,'_ses-','audio','_acq-',acq1,'_desc-',desc,'_T1w.nii');

% get FreeSurfer directory
surf_dir = strcat(deri_dir,'freesurfer/','sub-',sub,'_ses-','audio','/surf/');


%%% Step 1: import cortical surface %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% import both hemispheres
% for i = 1:numel(hemi)
%     spm_prf_import_surface(glm_dir, strc_img, surf_dir, hemi{i});
% end;


%%% Step 2: threshold contrast map %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify thresholding
corr1 = 'FWE';  % corr1 = 'unc';
p     = 0.05;   % p     = 0.001;
k     = 10;     % k     = 10;
if strcmp(corr1,'unc'), corr2 = 'none'; else, corr2 = corr1; end;
suff  = strcat(corr1,'_',num2str(p),'_',num2str(k));

% threshold contrast
clear matlabbatch;
matlabbatch{1}.spm.stats.results.spmmat{1}           = strcat(glm_dir,'SPM.mat');
matlabbatch{1}.spm.stats.results.conspec.titlestr    = '';
matlabbatch{1}.spm.stats.results.conspec.contrasts   = 1;
matlabbatch{1}.spm.stats.results.conspec.threshdesc  = corr2;
matlabbatch{1}.spm.stats.results.conspec.thresh      = p;
matlabbatch{1}.spm.stats.results.conspec.extent      = k;
matlabbatch{1}.spm.stats.results.conspec.conjunction = 1;
matlabbatch{1}.spm.stats.results.conspec.mask.none   = 1;
matlabbatch{1}.spm.stats.results.units               = 1;
matlabbatch{1}.spm.stats.results.export{1}.binary.basename = suff;
spm_jobman('run',matlabbatch);

% restrict voxels to posterior part of the brain
% cd(glm_dir);
%  V        = spm_vol('spmF_0001_',suff,'.nii');
% [Y,XYZmm] = spm_read_vols(V);
%  i        = XYZmm(2,:) > 0;
%  Y(i)     = 0;
% spm_write_vol(V,Y);
% cd(dirs.tool_dir);


%%% Step 3: extract time series %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% extract all voxels
clear matlabbatch;
matlabbatch{1}.spm.util.voi.spmmat{1}             = strcat(glm_dir,'SPM.mat');
matlabbatch{1}.spm.util.voi.adjust                = 1;
matlabbatch{1}.spm.util.voi.session               = Inf;
matlabbatch{1}.spm.util.voi.name                  = strcat('model-',model,'_run');
matlabbatch{1}.spm.util.voi.roi{1}.mask.threshold = 0.5;
matlabbatch{1}.spm.util.voi.roi{1}.mask.image{1}  = strcat(glm_dir,'spmF_0001_',suff,'.nii');
matlabbatch{1}.spm.util.voi.expression            = 'i1';
spm_jobman('run',matlabbatch);
cd(dirs.tool_dir);

% rename VOI mask file
file_pref = 'VOI_model-pRF_run';
for i = 1:numel(run)
    movefile(strcat(glm_dir,file_pref,'_',num2str(run(i)),'.mat'), ...
             strcat(glm_dir,file_pref,'-',num2str(run(i)),'_timeseries.mat'), 'f');
    movefile(strcat(glm_dir,file_pref,'_',num2str(run(i)),'_eigen.nii'), ...
             strcat(glm_dir,file_pref,'-',num2str(run(i)),'_eigen.nii'), 'f');
end;
movefile(strcat(glm_dir,file_pref,'_mask.nii'), ...
         strcat(glm_dir,file_pref,'-all_mask.nii'), 'f');

% extract both hemispheres
% for i = 1:numel(hemi)
%     clear matlabbatch;
%     matlabbatch{1}.spm.util.voi.spmmat{1}             = strcat(glm_dir,'SPM.mat');
%     matlabbatch{1}.spm.util.voi.adjust                = 1;
%     matlabbatch{1}.spm.util.voi.session               = Inf;
%     matlabbatch{1}.spm.util.voi.name                  = strcat(hemi{i},'_pRF_mask');
%   % matlabbatch{1}.spm.util.voi.roi{1}.mask.threshold = 0.5;
%     matlabbatch{1}.spm.util.voi.roi{1}.mask.image{1}  = strcat(glm_dir,'spmF_0001_',suff,'.nii');
%     matlabbatch{1}.spm.util.voi.roi{2}.mask.image{1}  = strcat(glm_dir,hemi{i},'_surface.nii');
%     matlabbatch{1}.spm.util.voi.expression            = 'i1 & i2';
%     spm_jobman('run',matlabbatch);
% end;