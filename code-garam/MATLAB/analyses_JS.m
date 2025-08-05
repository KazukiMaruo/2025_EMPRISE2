% Analyses of the EMPRISE data set
% by Joram Soch <soch@cbs.mpg.de>


clear

%%% Session EDY7-visual: BayespRF, 2023-09-11 #############################

% specify analysis
addpath(genpath('C:\spm\BayespRF\toolbox\'));
dirs     = load('project_directories.mat');
subj_id  = 'EDY7';
session  = 'visual';
mod_name = 'pRF';
ana_base = 'Analysis';
run      = [1:8];
avg      = [false, true];
conf     = [false, true];
voxels   = [];

% load confounds
create_mult_regs(subj_id, session, mod_name, 0);
mod_dir = strcat(dirs.data_dir,'derivatives/spm12/','sub-',subj_id,'/','ses-',session,'/','model-',mod_name,'/');
for i = 1:numel(run)
    filename = strcat(mod_dir,'sub-',subj_id,'_','ses-',session,'_','model-',mod_name,'_','run-',num2str(run(i)),'_','regressors.mat');
    load(filename);
    X_c(:,:,i) = [R, ones(size(R,1),1)];
end;
clear filename R

% analyze data
for i = 1:numel(avg)
    for j = 1:numel(conf)
        % specify target file
        ana_name = strcat(ana_base,'_',num2str(avg(i)),'_',num2str(conf(j)));
        filename = strcat(mod_dir,'PRF_',ana_name,'.mat');
        fprintf('\n-> ESTIMATION VARIANT: avg=%s, conf=%s:', num2str(avg(i)), num2str(conf(j)));
        % if file already exists, report this
        if exist(filename,'file')
            fprintf('\n   - has already been run and saved to disk!\n');
        % if file does not exist, run analysis
        else
            fprintf('\n   - has not been saved to disk and will now be run:\n')
            if ~conf(j)
                BpRF_run_pRF_analysis(subj_id, session, mod_name, ana_name, avg(i), [], voxels);
            else
                BpRF_run_pRF_analysis(subj_id, session, mod_name, ana_name, avg(i), X_c, voxels);
            end;
        end;
    end;
end;


%%% Simulation C: BayespRF, 2023-09-07 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % specify analysis
% addpath(genpath('C:\spm\BayespRF\toolbox\'));
% sim_dir  = 'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulated_data\';
% mod_dir  = 'C:\Joram\projects\MPI\EMPRISE\data\derivatives\spm12\sub-EDY7\ses-visual\model-pRF\';
% subj_id  = 'EDY7';
% session  = 'visual';
% mod_name = 'pRF';
% ana_name = 'Simulation_C';
% sim_file = strcat(sim_dir,ana_name,'.mat');
% avg      = [false, true];
% conf     = [false, true];
% voxels   = [];
% 
% % perform simulation
% for i = 1:numel(avg)
%     for j = 1:numel(conf)
%         % specify target file
%         filename = strcat(mod_dir,'PRF_',ana_name,'_',num2str(avg(i)),'_',num2str(conf(j)),'.mat');
%         fprintf('\n-> ESTIMATION VARIANT: avg=%s, conf=%s:', num2str(avg(i)), num2str(conf(j)));
%         % if file already exists, report this
%         if exist(filename,'file')
%             fprintf('\n   - has already been run and saved to disk!\n');
%         % if file does not exist, run analysis
%         else
%             fprintf('\n   - has not been saved to disk and will now be run:\n')
%             BpRF_run_pRF_simulation(subj_id, session, mod_name, ana_name, sim_file, avg(i), conf(j), voxels);
%         end;
%     end;
% end;