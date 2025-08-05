function create_onset_files(subj_id, session, mod_name, do_plot)
% _
% Create "names, onsets, durations" files for first-level fMRI analysis
% FORMAT create_onset_files(subj_id, session, mod_name, do_plot)
% 
%     subj_id  - a string, subject ID (e.g. "EDY7")
%     session  - a string, session name (e.g. "visual")
%     mod_name - a string, model name (e.g. "base")
%     do_plot  - a logical, indicating whether to visualize or
%                a scalar, indicating how much to visualize (e.g. 0.5)
% 
% FORMAT create_onset_files(subj_id, session, mod_name, do_plot) extracts
% the experimental conditions for selected subject, session and model and
% saves them nto an SPM-compatible "names, onsets, durations" file. If
% do_plot is true, then stimulus functions are shown for the first run.
% 
% Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
% 2023-05-17, 11:28: first version
% 2023-05-22, 10:24: adapted visualization
% 2023-05-25, 20:35: changed derivatives directory
% 2023-06-08, 14:48: introduced model name
% 2023-06-15, 15:58: implemented model "para"
% 2023-07-04, 10:31: transformed events to blocks
% 2023-07-04, 14:03: implemented model "pRF"


%%% Step 1: load logfiles %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set plotting, if necessary
if nargin < 3 || isempty(do_plot)
    do_plot = 0;
end;

% load project directories
dirs = load('project_directories.mat');

% specify numbers
num_runs = 8;
num_cons = 6;

% specify IDs
sub  = subj_id;
ses  = session;
task = 'harvey';
acq  = 'fMRI1p75TE24TR2100iPAT3FS';
run  = [1:num_runs];
model= mod_name;

% specify fMRI
n    = 145;
TR   = 2.1;
dt   = 0.01;

% load logfiles
events = cell(num_runs,1);
for i = 1:num_runs
    filename = strcat(dirs.data_dir,'sub-',sub,'/','ses-',ses,'/func/',...
                      'sub-',sub,'_','ses-',ses,'_','task-',task,'_','acq-',acq,'_','run-',num2str(run(i)),'_','events.tsv');
    [events{i}, hdr] = tsvload(filename);
end;


%%% Step 2: create onsets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% transform to blocks
blocks = cell(size(events));
for i = 1:num_runs
    blocks{i} = [];
    for j = 1:size(events{i},1)
        if j == 1
            blocks{i}        = [blocks{i}; events{i}(j,:)];
        elseif j == size(events{i},1)
          % blocks{i}{end,2} = (events{i}{j,1}+events{i}{j,2}) - blocks{i}{end,1};
            blocks{i}{end,2} = max(cell2mat(blocks{i}(1:end-1,2)));
        elseif ~strcmp(events{i}{j,3},events{i}{j-1,3})
          % blocks{i}{end,2} = (events{i}{j-1,1}+events{i}{j-1,2}) - blocks{i}{end,1};
            blocks{i}{end,2} = events{i}{j,1} - blocks{i}{end,1};
            blocks{i}        = [blocks{i}; events{i}(j,:)];
        end;
    end;
end;

% models "base" & "pRF"
%-------------------------------------------------------------------------%
if strncmp(model,'base',4) || strcmp(model,'pRF')

% specify names
names = {'one', 'two', 'three', 'four', 'five', 'many'};
if strcmp(ses,'visual')
    labels = {'1_dot', '2_dot', '3_dot', '4_dot', '5_dot', '20_dot'};
elseif strcmp(ses,'audio')
    labels = {'1_audio', '2_audio', '3_audio', '4_audio', '5_audio', '20_audio'};
end;

% collect onsets
for i = 1:num_runs
    
    % extract onsets and durations
    onsets    = cell(1,num_cons);
    durations = cell(1,num_cons);
    if strcmp(model,'base')
        for j = 1:num_cons
            onsets{j}    = cell2mat(events{i}(strcmp(events{i}(:,3),labels{j}),1));
            durations{j} = cell2mat(events{i}(strcmp(events{i}(:,3),labels{j}),2));
        end;
    elseif strcmp(model,'base-v2') || strcmp(model,'pRF')
        for j = 1:num_cons
            onsets{j}    = cell2mat(blocks{i}(strcmp(blocks{i}(:,3),labels{j}),1));
            durations{j} = cell2mat(blocks{i}(strcmp(blocks{i}(:,3),labels{j}),2));
        end;
    end;
    
    % create stimulus functions
    if i == 1
        T  = n*TR;
        t  = [0:dt:(T-dt)];
        SF = zeros(num_cons,numel(t));
        for j = 1:num_cons
            for k = 1:numel(onsets{j})
                SF(j, t>=onsets{j}(k) & t<(onsets{j}(k)+durations{j}(k))) = 1;
            end;
        end;
    end;
    
    % save names/onsets/durations
    deri_dir  = strcat(dirs.data_dir,'derivatives/spm12/');
    if ~exist(deri_dir,'dir'), mkdir(deri_dir); end;
    subj_dir = strcat(deri_dir,'sub-',sub,'/');
    if ~exist(subj_dir,'dir'), mkdir(subj_dir); end;
    sess_dir = strcat(subj_dir,'ses-',ses,'/');
    if ~exist(sess_dir,'dir'), mkdir(sess_dir); end;
    mod_dir  = strcat(sess_dir,'model-',model,'/');
    if ~exist(mod_dir,'dir'), mkdir(mod_dir); end;
    filename = strcat(mod_dir,'sub-',sub,'_','ses-',ses,'_','model-',model,'_','run-',num2str(run(i)),'_','onsets.mat');
    save(filename, 'names', 'onsets', 'durations');
    
end;

end;

% models "para"
%-------------------------------------------------------------------------%
if strncmp(model,'para',4)

% specify labels
if strcmp(ses,'visual')
    labels = {'1_dot', '2_dot', '3_dot', '4_dot', '5_dot', '20_dot'};
elseif strcmp(ses,'audio')
    labels = {'1_audio', '2_audio', '3_audio', '4_audio', '5_audio', '20_audio'};
end;

% collect onsets
for i = 1:num_runs
    
    % extract onsets and durations
    names     = {'few', 'many'};
    onsets    = cell(1,numel(names));
    durations = cell(1,numel(names));
    pmod(1).name{1}  = 'numerosity';
    pmod(1).poly{1}  = 1;
    pmod(1).param{1} = [];
    if strcmp(model,'para')
        onsets{1}    = cell2mat(blocks{i}(~strcmp(blocks{i}(:,3),labels{end}),1));
        onsets{2}    = cell2mat(blocks{i}(strcmp(blocks{i}(:,3),labels{end}),1));
        durations{1} = cell2mat(blocks{i}(~strcmp(blocks{i}(:,3),labels{end}),2));
        durations{2} = cell2mat(blocks{i}(strcmp(blocks{i}(:,3),labels{end}),2));
        for j = 1:size(blocks{i},1)
            l = find(strcmp(labels,blocks{i}{j,3}));
            if l < 6, pmod(1).param{1} = [pmod(1).param{1}, l]; end;
        end;
    end;
    pmod(1).param{1} = pmod(1).param{1} - 3;
    orth = num2cell(false(size(names)));
    
    % save names/onsets/durations
    deri_dir  = strcat(dirs.data_dir,'derivatives/spm12/');
    if ~exist(deri_dir,'dir'), mkdir(deri_dir); end;
    subj_dir = strcat(deri_dir,'sub-',sub,'/');
    if ~exist(subj_dir,'dir'), mkdir(subj_dir); end;
    sess_dir = strcat(subj_dir,'ses-',ses,'/');
    if ~exist(sess_dir,'dir'), mkdir(sess_dir); end;
    mod_dir  = strcat(sess_dir,'model-',model,'/');
    if ~exist(mod_dir,'dir'), mkdir(mod_dir); end;
    filename = strcat(mod_dir,'sub-',sub,'_','ses-',ses,'_','model-',model,'_','run-',num2str(run(i)),'_','onsets.mat');
    save(filename, 'names', 'onsets', 'durations', 'pmod', 'orth');
    
end;

end;


%%% Step 3: visualize onsets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% models "base"
if strncmp(model,'base',4)

% plot stimulus functions
if do_plot > 0
    figure('Name', sprintf('EMPRISE: Subject "%s", Session "%s", Run 1', sub, ses), 'Color', [1 1 1], 'Position', [50 50 1600 900]);
    for j = 1:num_cons
        subplot(num_cons, 1, j);
        plot(t, SF(j,:), '-b', 'LineWidth', 1);
        axis([0, do_plot*T, (0-0.1), (1+0.1)]);
        set(gca,'XTick',[0:25:T]);
        set(gca,'YTick',[0, 1]);
        if j == num_cons, xlabel('time [s]', 'FontSize', 16); end;
        ylabel(names{j}, 'FontSize', 16);
        if j == 1, title('EMPRISE: stimulus functions', 'FontSize', 16); end;
    end;
end;

end;