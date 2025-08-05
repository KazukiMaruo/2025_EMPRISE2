function create_mult_regs(subj_id, session, mod_name, do_plot)
% _
% Create "multiple regressors" files for first-level fMRI analysis
% FORMAT create_mult_regs(subj_id, session, mod_name, do_plot)
% 
%     subj_id - a string, subject ID (e.g. "EDY7")
%     session - a string, session name (e.g. "visual")
%     mod_name   - a string, model name (e.g. "base")
%     do_plot - a logical, indicating whether to visualize or
%               a scalar, indicating how much to visualize (e.g. 0.5)
% 
% FORMAT create_onset_files(subj_id, session, mod_name, do_plot) extracts
% the confounds timeseries for selected subject, session and model and
% saves them into an SPM-compatible "multiple regressors" file. If do_plot
% is true, then those timeseries are shown for the first run.
% 
% Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
% 2023-05-25, 11:42: first version
% 2023-05-25, 20:36: changed derivatives directory
% 2023-06-08, 13:46: introduced model name
% 2023-06-15, 16:00: implemented model "para"
% 2023-07-04, 14:07: implemented model "pRF"


%%% Step 1: load confounds %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set plotting, if necessary
if nargin < 3 || isempty(do_plot)
    do_plot = 0;
end;

% load project directories
dirs = load('project_directories.mat');

% specify numbers
num_runs = 8;

% specify IDs
sub  = subj_id;
ses  = session;
task = 'harvey';
acq  = 'fMRI1p75TE24TR2100iPAT3FS';
run  = [1:num_runs];
desc = 'confounds';
model= mod_name;

% specify fMRI
n    = 145;
TR   = 2.1;

% load confounds
covs = cell(num_runs,1);
for i = 1:num_runs
    filename = strcat(dirs.prep_dir,'sub-',sub,'/','ses-',ses,'/func/',...
                      'sub-',sub,'_','ses-',ses,'_','task-',task,'_','acq-',acq,'_',...
                      'run-',num2str(run(i)),'_','desc-',desc,'_','timeseries.tsv');
    [covs{i}, hdr{i}] = tsvload(filename);
end;


%%% Step 2: create regressors %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% models "base" & "para"
if strncmp(model,'base',4) || strcmp(model,'para') || strcmp(model,'pRF')

% specify names
labels= {'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'white_matter', 'csf', 'global_signal'};
names = {'trans-x', 'trans-y', 'trans-z', 'rot-x', 'rot-y', 'rot_z', 'wm-signal', 'csf-signal', 'global-signal'};

% collect confounds
for i = 1:num_runs
    
    % extract multiple regressors
    R = zeros(n,numel(names));
    for j = 1:numel(names)
        R(:,j) = cell2mat(covs{i}(:,strcmp(hdr{i},labels{j})));
        if strncmp(labels{j},'rot_',4)      % convert rotations
            R(:,j) = (180/pi)*R(:,j);       % from [rad] to [deg]
        end;
        if strcmp(labels{j},'white_matter') || strcmp(labels{j},'csf') || strcmp(labels{j},'global_signal')
            R(:,j) = R(:,j) - mean(R(:,j)); % subtract arithmetic mean
            R(:,j) = R(:,j)./ std(R(:,j));  % divide by standard deviation
        end;
    end;
    
    % create plotting regressors
    if i == 1
        T   = n*TR;                         % total scanning time
        t   =[0:TR:((n-1)*TR)];             % fMRI acquisition onsets
        R1a = R(:,1:6);                     % translations & rotations
        R1b = R(:,7:9);                     % average signals
        if strcmp(model,'base-v2')
            R1b = R(:,1:end-1);
        end;
    end;
    
    % save multiple regressors
    deri_dir  = strcat(dirs.data_dir,'derivatives/spm12/');
    if ~exist(deri_dir,'dir'), mkdir(deri_dir); end;
    subj_dir = strcat(deri_dir,'sub-',sub,'/');
    if ~exist(subj_dir,'dir'), mkdir(subj_dir); end;
    sess_dir = strcat(subj_dir,'ses-',ses,'/');
    if ~exist(sess_dir,'dir'), mkdir(sess_dir); end;
    mod_dir  = strcat(sess_dir,'model-',model,'/');
    if ~exist(mod_dir,'dir'), mkdir(mod_dir); end;
    filename = strcat(mod_dir,'sub-',sub,'_','ses-',ses,'_','model-',model,'_','run-',num2str(run(i)),'_','regressors.mat');
    save(filename, 'names', 'R');
    if strcmp(model,'base-v2')              % base V2: remove global signal
        names = names(1:end-1);
        R     = R(:,1:end-1);
        save(filename, 'names', 'R');
    end;
    
end;

end;


%%% Step 3: visualize confounds %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% models "base"
if strncmp(model,'base',4)

% plot mulitple regressors
if do_plot > 0
    figure('Name', sprintf('EMPRISE: Subject "%s", Session "%s", Run 1', sub, ses), 'Color', [1 1 1], 'Position', [50 50 1600 900]);
    
    subplot(2,1,1);
    plot(t', R1a);
    axis([0, do_plot*T, -(11/10)*max(max(abs(R1a))), +(11/10)*max(max(abs(R1a)))]);
    set(gca,'XTick',[0:25:T]);
    legend(names(1:6), 'Location', 'SouthWest');
    xlabel('time [s]', 'FontSize', 16);
    ylabel('translations [mm], rotations [deg]', 'FontSize', 16);
    title('Translations & Rotations', 'FontSize', 16);
    
    subplot(2,1,2);
    plot(t', R1b);
    rng1b = max(max(R1b))-min(min(R1b));
    axis([0, do_plot*T, min(min(R1b))-(1/10)*rng1b, max(max(R1b))+(1/10)*rng1b]);
    set(gca,'XTick',[0:25:T]);
    legend(names(7:end), 'Location', 'NorthEast');
    xlabel('time [s]', 'FontSize', 16);
    ylabel('mean-centered signals [a.u.]', 'FontSize', 16);
    title('Average Signals', 'FontSize', 16);
end;

end;