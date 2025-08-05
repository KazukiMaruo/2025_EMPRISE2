% Analyses of the EMPRISE data set
% by Joram Soch <soch@cbs.mpg.de>


clear

%%% BayespRF empirical analysis, 2023-08-24 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % set subject and session
% subj_id = 'EDY7';
% sess_id = 'visual';
% 
% % estimate first-level GLM
% create_onset_files(subj_id, sess_id, 0);
% create_mult_regs(subj_id, sess_id, 0);
% create_stats_batch(subj_id, sess_id, 'pRF', true);
% 
% % run voxel-wise BayespRF analysis
% BpRF_run_first_level(subj_id, sess_id, 'pRF');
% BpRF_run_pRF_analysis(subj_id, session, 'pRF', []);


%%% BayespRF simulation analysis, 2023-08-17/21 %%%%%%%%%%%%%%%%%%%%%%%%%%%

% % specify analysis
% addpath(genpath('C:\spm\BayespRF\toolbox\'));
% sim_dir  = 'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulated_data\';
% subj_id  = 'EDY7';
% session  = 'visual';
% mod_name = 'pRF';
% ana_name ={'Simulation_A', 'Simulation_B'};
% sim_file = strcat(sim_dir,ana_name{1},'.mat');
% sim_file = strcat(sim_dir,ana_name{2},'.mat');
% voxels   = []; % [1:3];
% 
% % perform simulation
% BpRF_run_pRF_simulation(subj_id, session, mod_name, ana_name{1}, sim_file, voxels);
% BpRF_run_pRF_simulation(subj_id, session, mod_name, ana_name{2}, sim_file, voxels);
% 
% % load true parameters (Simulation)
% Sim = load(sim_file);
% mu_true   = Sim.mu;
% fwhm_true = Sim.fwhm;
% clear Sim
% 
% % load estimated parameters (NumpRF)
% load(strcat(sim_dir,'Simulation_B_NumpRF_avg_True_False.mat'));
% mu_mle   = mu_est;
% fwhm_mle = fwhm_est;
% clear mu_est fwhm_est time
% 
% % load estimated parameters (BayespRF)
% mod_dir  = 'C:\Joram\projects\MPI\EMPRISE\data\derivatives\spm12\sub-EDY7\ses-visual\model-pRF\';
% load(strcat(mod_dir,'PRF_',ana_name{1},'.mat'));
% load(strcat(mod_dir,'PRF_',ana_name{2},'.mat'));
% if isempty(voxels), voxels = [1:numel(PRF.Ep)]; end;
% mu_est   = zeros(size(voxels));
% fwhm_est = zeros(size(voxels));
% for i = 1:numel(voxels)
%     P = spm_prf_fcn_numerosity(PRF.Ep{voxels(i)}, PRF.M, PRF.U, 'get_summary');
%     mu_est(i)   = P.mu;
%     fwhm_est(i) = P.fwhm;
% end;
% 
% % plot estimated parameters (BayespRF)
% figure('Name', 'BayespRF', 'Color', [1 1 1], 'Position', [50 50 800 800]);
% plot(mu_est, fwhm_est, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([0, 10, 0, 10]);
% axis square
% xlabel('estimated preferred numerosity', 'FontSize', 16);
% ylabel('estimated tuning width', 'FontSize', 16);
% title('BayespRF simulation', 'FontSize', 24);
% 
% % plot estimated parameters (BayespRF vs. truth)
% figure('Name', 'BayespRF', 'Color', [1 1 1], 'Position', [50 50 1600 800]);
% subplot(1,2,1);
% plot(mu_true, mu_est, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([0, 10, 0, 10]);
% axis square
% xlabel('ground truth value', 'FontSize', 16);
% ylabel('BayespRF estimate', 'FontSize', 16);
% title('preferred numerosity', 'FontSize', 24);
% subplot(1,2,2);
% plot(fwhm_true, fwhm_est, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([0, 10, 0, 10]);
% axis square
% xlabel('ground truth value', 'FontSize', 16);
% ylabel('BayespRF estimate', 'FontSize', 16);
% title('tuning width', 'FontSize', 24);
% 
% % plot estimated parameters (NumpRF)
% figure('Name', 'NumpRF', 'Color', [1 1 1], 'Position', [50 50 800 800]);
% plot(mu_mle, fwhm_mle, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([0, 10, 0, 10]);
% axis square
% xlabel('estimated preferred numerosity', 'FontSize', 16);
% ylabel('estimated tuning width', 'FontSize', 16);
% title('NumpRF simulation', 'FontSize', 24);
% 
% % plot estimated parameters (NumpRF vs. truth)
% figure('Name', 'NumpRF', 'Color', [1 1 1], 'Position', [50 50 1600 800]);
% subplot(1,2,1);
% plot(mu_true, mu_mle, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([0, 10, 0, 10]);
% axis square
% xlabel('ground truth value', 'FontSize', 16);
% ylabel('NumpRF estimate', 'FontSize', 16);
% title('preferred numerosity', 'FontSize', 24);
% subplot(1,2,2);
% plot(fwhm_true, fwhm_mle, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([0, 10, 0, 10]);
% axis square
% xlabel('ground truth value', 'FontSize', 16);
% ylabel('NumpRF estimate', 'FontSize', 16);
% title('tuning width', 'FontSize', 24);


%%% BayespRF/NumpRF simulation results, 2023-08-14 %%%%%%%%%%%%%%%%%%%%%%%%

% % load estimated parameters (BayespRF)
% dirs    = load('project_directories.mat');
% mod_dir = strcat(dirs.data_dir,'derivatives/','spm12/','sub-','EDY7','/','ses-','visual','/','model-','pRF','/');
% load(strcat(mod_dir,'PRF_simulation.mat'));
% voxels  = [1:numel(PRF.Ep)];
% mu_est  = zeros(size(voxels));
% fwhm_est= zeros(size(voxels));
% for i = 1:numel(voxels)
%     P = spm_prf_fcn_numerosity(PRF.Ep{voxels(i)}, PRF.M, PRF.U, 'get_summary');
%     mu_est(i)   = P.mu;
%     fwhm_est(i) = P.fwhm;
% end;
% 
% % load estimated parameters (NumpRF)
% load(strcat(dirs.tool_dir,'../../../Python/simulated_data/','Simulation_B.mat'));
% mu_true   = mu;
% fwhm_true = fwhm;
% close all
% 
% % plot estimated parameters (BayespRF A)
% figure('Name', 'BayespRF', 'Color', [1 1 1], 'Position', [50 50 800 800]);
% plot(mu_est, fwhm_est, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([0, 9, 0, 9]);
% axis square
% xlabel('estimated preferred numerosity', 'FontSize', 16);
% ylabel('estimated tuning width', 'FontSize', 16);
% title('BayespRF simulation', 'FontSize', 24);
% 
% % plot estimated parameters (BayespRF B)
% figure('Name', 'BayespRF', 'Color', [1 1 1], 'Position', [50 50 800 800]);
% plot(mu_est, fwhm_est, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([0, 15, 0, 15]);
% axis square
% xlabel('estimated preferred numerosity', 'FontSize', 16);
% ylabel('estimated tuning width', 'FontSize', 16);
% title('BayespRF simulation', 'FontSize', 24);
% 
% % plot estimated parameters (BayespRF vs. truth)
% figure('Name', 'BayespRF', 'Color', [1 1 1], 'Position', [50 50 1600 800]);
% subplot(1,2,1);
% plot(mu_true, mu_est, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([0, 10, 0, 10]);
% axis square
% xlabel('ground truth value', 'FontSize', 16);
% ylabel('BayespRF estimate', 'FontSize', 16);
% title('preferred numerosity', 'FontSize', 24);
% subplot(1,2,2);
% plot(fwhm_true, fwhm_est, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([0, 10, 0, 10]);
% axis square
% xlabel('ground truth value', 'FontSize', 16);
% ylabel('BayespRF estimate', 'FontSize', 16);
% title('tuning width', 'FontSize', 24);


%%% BayespRF simulation analysis, 2023-08-10 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % specify analyses
% addpath(genpath('C:\spm\BayespRF\toolbox\'));
% subj_id  = 'EDY7';
% session  = 'visual';
% mod_name = 'pRF';
% voxels   = [];
% sim_file = 'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulated_data\Simulation_B.mat';
% 
% % estimate population receptive field model
% BpRF_run_pRF_simulation(subj_id, session, mod_name, sim_file, voxels);


%%% BayespRF demo analysis, 2023-08-07 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % specify analyses
% addpath(genpath('C:\spm\BayespRF\toolbox\'));
% subj_id  = 'EDY7';
% session  = 'visual';
% mod_name = 'pRF';
% voxels   = [];
% 
% % estimate population receptive field model
% BpRF_run_pRF_demo(subj_id, session, mod_name, voxels);


%%% BayespRF/NumpRF empirical results, 2023-08-07 %%%%%%%%%%%%%%%%%%%%%%%%%

% % load estimated parameters (BayespRF)
% dirs    = load('project_directories.mat');
% mod_dir = strcat(dirs.data_dir,'derivatives/','spm12/','sub-','EDY7','/','ses-','visual','/','model-','pRF','/');
% load(strcat(mod_dir,'PRF_numerosity.mat'));
% voxels= [1:numel(PRF.Ep)];
% muB   = zeros(size(voxels));
% fwhmB = zeros(size(voxels));
% for i = 1:numel(voxels)
%     P = spm_prf_fcn_numerosity(PRF.Ep{voxels(i)}, PRF.M, PRF.U, 'get_summary');
%     muB(i)   = P.mu;
%     fwhmB(i) = P.fwhm;
% end;
% 
% % load estimated parameters (NumpRF)
% load(strcat(mod_dir,'NumpRF.mat'));
% muC   = mu_est;
% fwhmC = fwhm_est;
% 
% % plot estimated parameters (BayespRF)
% figure('Name', 'BayespRF', 'Color', [1 1 1], 'Position', [50 50 800 800]);
% plot(muB, fwhmB, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([0, 15, 0, 15]);
% axis square
% xlabel('estimated preferred numerosity', 'FontSize', 16);
% ylabel('estimated tuning width', 'FontSize', 16);
% title('BayespRF analysis', 'FontSize', 24);
% 
% % plot estimated parameters (BayespRF vs. NumpRF)
% figure('Name', 'BayespRF vs. NumpRF', 'Color', [1 1 1], 'Position', [50 50 1600 800]);
% subplot(1,2,1); hold on;
% plot([0, 15], [0, 15], '-k', 'LineWidth', 2);
% plot(muC, muB, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([0, 15, 0, 15]);
% axis square
% set(gca,'Box','On');
% xlabel('maximum likelihood estimation (NumpRF)', 'FontSize', 16);
% ylabel('Bayesian inference (BayespRF)', 'FontSize', 16);
% title('preferred numerosity', 'FontSize', 24);
% subplot(1,2,2); hold on;
% plot([0, 15], [0, 15], '-k', 'LineWidth', 2);
% plot(fwhmC, fwhmB, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([0, 15, 0, 15]);
% axis square
% set(gca,'Box','On');
% xlabel('maximum likelihood estimation (NumpRF)', 'FontSize', 16);
% ylabel('Bayesian inference (BayespRF)', 'FontSize', 16);
% title('tuning width', 'FontSize', 24);


%%% BayespRF simulation analysis, 2023-07-13 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % specify analyses
% addpath(genpath('C:\spm\BayespRF\toolbox\'));
% subj_id  = 'EDY7';
% session  ={'visual', 'audio'};
% mod_name = 'pRF';
% voxels   = [1:5]; % [];
% sim_file = 'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulated_data\Simulation_B.mat';
% 
% % estimate population receptive field model
% BpRF_run_pRF_simulation(subj_id, session{1}, mod_name, sim_file, voxels);
% 
% % review estimated pRF model parameters
% Sim     = load(sim_file);
% dirs    = load('project_directories.mat');
% PRF_mat = strcat(dirs.data_dir,'derivatives/','spm12/','sub-',subj_id,'/','ses-',session{1},'/','model-',mod_name,'/','PRF_simulation.mat');
% load(PRF_mat);
% % voxels= [1:numel(PRF.Ep)];
% mu    = zeros(size(voxels));
% fwhm  = zeros(size(voxels));
% beta  = zeros(size(voxels));
% Sim_b = mean(Sim.B(1,:,:),3);
% for i = 1:numel(voxels)
%     P = spm_prf_fcn_numerosity(PRF.Ep{voxels(i)}, PRF.M, PRF.U, 'get_summary');
%     mu(i)   = P.mu;
%     fwhm(i) = P.fwhm;
%     beta(i) = P.beta;
% end;
% figure('Name', 'NumpRF', 'Color', [1 1 1], 'Position', [50 50 1800 600]);
% subplot(1,3,1);
% plot(Sim.mu(voxels), mu, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% subplot(1,3,2);
% plot(Sim.fwhm(voxels), fwhm, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% subplot(1,3,3);
% plot(Sim_b(voxels), beta, 'ob', 'LineWidth', 2, 'MarkerSize', 5);


%%% BayespRF empirical analysis, 2023-07-04/06 %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % specify analyses
% addpath(genpath('C:\spm\BayespRF\toolbox\'));
% subj_id  = 'EDY7';
% session  ={'visual', 'audio'};
% mod_name = 'pRF';
% voxels   = []; % [100:100:700];
% 
% % extract time series from first-level model
% % BpRF_run_first_level(subj_id, session{1}, mod_name);
% 
% % estimate population receptive field model
% % BpRF_run_pRF_analysis(subj_id, session{1}, mod_name, voxels);
% 
% % review estimated pRF model parameters
% dirs    = load('project_directories.mat');
% PRF_mat = strcat(dirs.data_dir,'derivatives/','spm12/','sub-',subj_id,'/','ses-',session{1},'/','model-',mod_name,'/','PRF_numerosity.mat');
% load(PRF_mat);
% voxels= [1:numel(PRF.Ep)];
% mu    = zeros(size(voxels));
% fwhm  = zeros(size(voxels));
% for i = 1:numel(voxels)
%     P = spm_prf_fcn_numerosity(PRF.Ep{voxels(i)}, PRF.M, PRF.U, 'get_summary');
%     mu(i)   = P.mu;
%     fwhm(i) = P.fwhm;
% end;
% figure('Name', 'NumpRF', 'Color', [1 1 1], 'Position', [50 50 800 800]);
% plot(mu, fwhm, 'ob', 'LineWidth', 2, 'MarkerSize', 5);
% axis([1, 20, 1, 20]);


%%% SPM first-level analyses (pRF), 2023-07-04 %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % specify analyses
% subj_id  = 'EDY7';
% session  ={'visual', 'audio'};
% mod_name = 'pRF';
% 
% % perform analyses
% for i = 1:numel(session)
%     create_onset_files(subj_id, session{i}, mod_name, 0.5);
%     create_mult_regs(subj_id, session{i}, mod_name, false);
%     create_stats_batch(subj_id, session{i}, mod_name, true);
% end;


%%% mcheck test analysis, 2023-06-26 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % specify analysis
% data = 'raw';
% tool = 'mcheck';
% 
% % specify IDs
% if strcmp(data,'raw')
%     data_dir = 'C:\Joram\projects\MPI\EMPRISE\data\';
% elseif strcmp(data,'preproc')
%     data_dir = 'C:\Joram\projects\MPI\EMPRISE\data\derivatives\fmriprep\';
% end;    
% sub  = 'EDY7';
% ses  = 'visual';
% task = 'harvey';
% acq  = 'fMRI1p75TE24TR2100iPAT3FS';
% run  = [1:8];
% space= 'T1w';
% desc = 'preproc';
% 
% % specify images
% filenames = cell(numel(run),1);
% for i = 1:numel(run)
%     if strcmp(data,'raw')
%         filenames{i} = strcat(data_dir,'sub-',sub,'/','ses-',ses,'/func/',...
%                               'sub-',sub,'_','ses-',ses,'_','task-',task,'_','acq-',acq,'_',...
%                               'run-',num2str(run(i)),'_','bold.nii');
%     elseif strcmp(data,'preproc')
%         filenames{i} = strcat(data_dir,'sub-',sub,'/','ses-',ses,'/func/',...
%                               'sub-',sub,'_','ses-',ses,'_','task-',task,'_','acq-',acq,'_',...
%                               'run-',num2str(run(i)),'_','space-',space,'_','desc-',desc,'_','bold.nii');
%     end;
% end;
% 
% % check functional images (mcheck)
% if strcmp(tool,'mcheck')
%     % load images
%     num_vol = 0;
%     sy      = zeros(1,numel(run)+1);
%     sy(1)   = num_vol + 1;
%     for i = 1:numel(run)
%         H = spm_vol(filenames{i});
%         Y = spm_read_vols(H);
%       % S(i).row   = [(num_vol+1):(num_vol+numel(H))];
%         S(i).sess  = i*ones(numel(H),1);
%         S(i).Y     = reshape(Y,[prod(H(1).dim), numel(H)])';
%         num_vol    = num_vol + numel(H);
%         sy(i+1)    = num_vol + 1;
%     end;
% 
%     % check functional images
%     X = vertcat(S.Y);
%     Y = vertcat(S.sess);
%     X = reshape(X,[size(X,1), size(X,2)/H(1).dim(3), H(1).dim(3)]);
%     X = permute(X,[3,2,1]);
%     addpath(strcat(pwd,'/other_tools/mcheck/'));
%     checkMotion(X, Y, sy);
% end;
% 
% % check functional images (JS)
% if strcmp(tool,'JS')
%     check_func_imgs(filenames);
% end;


%%% SPM first-level analyses (parametric), 2023-06-15 %%%%%%%%%%%%%%%%%%%%%

% create onset files
% create_onset_files('EDY7', 'visual', 'para', false);
% create_onset_files('EDY7', 'audio',  'para', false);

% create multiple regressors
% create_mult_regs('EDY7', 'visual', 'para', false);
% create_mult_regs('EDY7', 'audio',  'para', false);

% run statistics batches
% create_stats_batch('EDY7', 'visual', 'para', true);
% create_stats_batch('EDY7', 'audio',  'para', true);


%%% SPMfirst-level analyses (V2), 2023-06-08 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create onset files
% create_onset_files('EDY7', 'visual', 'base-v2', 0.5);
% create_onset_files('EDY7', 'audio',  'base-v2', 0.5);

% create multiple regressors
% create_mult_regs('EDY7', 'visual', 'base-v2', 0);
% create_mult_regs('EDY7', 'audio',  'base-v2', 0);

% run statistics batches
% create_stats_batch('EDY7', 'visual', 'base-v2', true);
% create_stats_batch('EDY7', 'audio',  'base-v2', true);


%%% VistaSoft analysis, 2023-05-30 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% add VistaSoft path
% addpath('VistaSoft_JS/');

% copy preprocessed data
% copy_preproc_data('EDY7', 'visual');
% copy_preproc_data('EDY7', 'audio');

% run numerosity model
% run_numerosity_model('EDY7', 'visual');
% run_numerosity_model('EDY7', 'audio');


%%% SPM first-level analyses, 2023-05-25 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % create onset files
% create_onset_files('EDY7', 'visual', 0);
% create_onset_files('EDY7', 'audio',  0);
% 
% % create multiple regressors
% create_mult_regs('EDY7', 'visual', 1);
% create_mult_regs('EDY7', 'audio',  1);
% 
% % run statistics batches
% create_stats_batch('EDY7', 'visual', 'base', true);
% create_stats_batch('EDY7', 'audio',  'base', true);


%%% pilot analysis: onset files, 2023-05-17 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % create onset files
% create_onset_files('EDY7', 'visual', 0.5);
% create_onset_files('EDY7', 'audio',  0.5);

% % specify IDs
% data_dir = 'C:\Joram\projects\MPI\EMPRISE\data\';
% sub  = 'EDY7';
% ses  = 'visual';
% task = 'harvey';
% acq  = 'fMRI1p75TE24TR2100iPAT3FS';
% run  = [1:8];
% 
% % specify fMRI
% n    = 145;
% TR   = 2.1;
% dt   = 0.01;
% 
% % load file
% filename = strcat(data_dir,'sub-',sub,'/','ses-',ses,'/func/',...
%                   'sub-',sub,'_','ses-',ses,'_','task-',task,'_','acq-',acq,'_','run-',num2str(run(1)),'_','events.tsv');
% [raw, hdr] = tsvload(filename);
% 
% % extract experimental conditions
% num_con   = 6;
% names     = {'one', 'two', 'three', 'four', 'five', 'many'};
% labels    = {'1_dot', '2_dot', '3_dot', '4_dot', '5_dot', '20_dot'};
% onsets    = cell(1,num_con);
% durations = cell(1,num_con);
% for j = 1:num_con
%     onsets{j}    = cell2mat(raw(strcmp(raw(:,3),labels{j}),1));
%     durations{j} = cell2mat(raw(strcmp(raw(:,3),labels{j}),2));
% end;
% 
% % create stimulus functions
% T  = n*TR;
% t  = [0:dt:(T-dt)];
% SF = zeros(num_con,numel(t));
% for j = 1:num_con
%     for k = 1:numel(onsets{j})
%         SF(j, t>=onsets{j}(k) & t<(onsets{j}(k)+durations{j}(k))) = 1;
%     end;
% end;
% 
% % visualize stimulus functions
% figure('Name', 'EMPRISE: Harvey task', 'Color', [1 1 1], 'Position', [50 50 1600 900]);
% 
% for j = 1:num_con
%     subplot(num_con, 1, j);
%     plot(t, SF(j,:), '-b', 'LineWidth', 1);
%     axis([0, T/2, (0-0.1), (1+0.1)]);
%     set(gca,'XTick',[0:25:T]);
%     set(gca,'YTick',[0, 1]);
%     if j == num_con, xlabel('time [s]', 'FontSize', 16); end;
%     ylabel(names{j}, 'FontSize', 16);
%     if j == 1, title('EMPRISE: stimulus functions', 'FontSize', 16); end;
% end;


%%% pilot analysis: functional images, 2023-05-17/22 %%%%%%%%%%%%%%%%%%%%%%

% % specify IDs
% data_dir = 'C:\Joram\projects\MPI\EMPRISE\data\derivatives\fmriprep\';
% sub  = 'EDY7';
% ses  = 'visual';
% task = 'harvey';
% acq  = 'fMRI1p75TE24TR2100iPAT3FS';
% run  = [1:8];
% space= 'T1w';
% desc = 'preproc';
% 
% % specify images
% filenames = cell(numel(run),1);
% for i = 1:numel(run)
%     filenames{i} = strcat(data_dir,'sub-',sub,'/','ses-',ses,'/func/',...
%                           'sub-',sub,'_','ses-',ses,'_','task-',task,'_','acq-',acq,'_',...
%                           'run-',num2str(run(i)),'_','space-',space,'_','desc-',desc,'_','bold.nii');
% end;
% 
% % check functional images
% check_func_imgs(filenames);

% % load images
% num_vol = 0;
% x_ticks = zeros(1,numel(run));
% for i = 1:numel(run)
%     H = spm_vol(filenames{i});
%     Y = spm_read_vols(H);
%     S(i).row   = [(num_vol+1):(num_vol+numel(H))];
%     S(i).Y     = reshape(Y,[prod(H(1).dim), numel(H)])';
%     x_ticks(i) = num_vol + 1;
%     num_vol    = num_vol + numel(H);
% end;
% clear H Y
% 
% % calculate correlations
% Y = vertcat(S.Y);
% R = corr(Y');
% 
% % average correlations
% r_in  = zeros(1,numel(run));
% r_out = zeros(1,numel(run));
% for i = 1:numel(run)
%     r_in(i)  = mean(mean(R(S(i).row,S(i).row)));
%     r_out(i) = mean(mean(R(horzcat(S(run~=i).row),S(i).row)));
% end;
% 
% % perform k-means
% num_cls = [2,3,4];
% idx     = zeros(numel(num_cls),num_vol);
% for i = 1:numel(num_cls)
%     idx(i,:) = kmeans(Y,num_cls(i))';
% end;
% 
% % visualize correlations
% figure('Name', 'EMPRISE: preprocessed fMRI', 'Color', [1 1 1], 'Position', [50 50 1200 800]);
% 
% subplot(2,2,1); hold on;
% cmap = colormap('jet');
% cmap = flipud(cmap);
% colormap(cmap);
% imagesc(R);
% caxis([0.75, 1]);
% axis([(1-0.5), (num_vol+0.5), (1-0.5), (num_vol+0.5)]);
% axis ij square;
% colorbar;
% set(gca,'XTick',x_ticks,'XTickLabels',cellstr(num2str(run'))');
% set(gca,'YTick',x_ticks,'YTickLabels',cellstr(num2str(run'))');
% xlabel('runs, volumes', 'FontSize', 12);
% ylabel('runs, volumes', 'FontSize', 12);
% title('all correlations', 'FontSize', 16);
% 
% subplot(2,2,2); hold on;
% plot(run, r_in,  '-og', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0,176,80]./255,  'MarkerFaceColor', [0,176,80]./255);
% plot(run, r_out, '-ob', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0,176,240]./255, 'MarkerFaceColor', [0,176,240]./255);
% axis([(1-0.5), (numel(run)+0.5), 0.75, 1]);
% set(gca,'Box','On');
% set(gca,'XTick',run);
% legend({'within runs', 'between runs'}, 'Location', 'SouthEast');
% xlabel('run', 'FontSize', 12);
% ylabel('correlation', 'FontSize', 12);
% title('mean correlations', 'FontSize', 16);
% 
% subplot(7,1,5);
% % cmap = [248, 251, 13; 169, 251, 87; 254, 183, 0]./255;
% % colormap(cmap);
% imagesc(idx(2,:));
% axis([(1-0.5), (num_vol+0.5), 0.5, 1.5]);
% set(gca,'XTick',x_ticks,'XTickLabels',cellstr(num2str(run'))');
% set(gca,'YTick',[0, 2]);
% ylabel('k = 3', 'FontSize', 12);
% 
% subplot(7,1,7);
% % cmap = [248, 251, 13; 254, 183, 0]./255;
% % colormap(cmap);
% imagesc(idx(1,:));
% axis([(1-0.5), (num_vol+0.5), 0.5, 1.5]);
% set(gca,'XTick',x_ticks,'XTickLabels',cellstr(num2str(run'))');
% set(gca,'YTick',[0, 2]);
% xlabel('runs, volumes', 'FontSize', 12);
% ylabel('k = 2', 'FontSize', 12);