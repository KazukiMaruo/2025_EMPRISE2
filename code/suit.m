addpath(genpath('/data/u_kazuki_software/EMPRISE_2/matlab/spm/'));
% addpath(genpath('/data/u_kazuki_software/EMPRISE_2/matlab/spm/toolbox/suit/'));

rowpath = '/data/u_kazuki_software/EMPRISE_2/suit/';
sess    = 'ses-visual';
model   = 'model-VolumetricFWHM3';
type    = 'anat';
param   = 'Rsq';

d = dir(fullfile(rowpath, 'sub-*'));
subject_list = {d([d.isdir]).name};  % Only keep directory names

%% Segmentation
for i = 1:length(subject_list)
    sub = subject_list{i}
    filename = [sub '_ses-visual_acq-bfcmprageised_T1w.nii'];
    
    fullpath = fullfile(rowpath, sub);
    if ~exist(fullpath,'dir')
        mkdir(fullpath);
    end
    cd(fullpath)
    
    %%% Isolation
    % suit_isolate_seg({filename});
    
    %%% Normalization
    % Find index of '.nii'
    % idx = strfind(filename, '.nii');
    
    %%% Insert string before it
    % fn_gray  = [filename(1:idx-1) '_seg1' filename(idx:end)];
    % fn_white = [filename(1:idx-1) '_seg2' filename(idx:end)];
    % fn_isol  = ['c_' filename(1:idx-1) '_pcereb' filename(idx:end)];
    % 
    % job.subjND.gray={fn_gray};
    % job.subjND.white={fn_white};
    % job.subjND.isolation={fn_isol};
    % 
    % suit_normalize_dartel(job)
    
    %%% Native Functional image to SUIT space
    % Find index of '.nii'
    idx = strfind(filename, '.nii');
    
    folder = fullfile(fullpath, sess, model);
    if ~exist(folder,'dir')
        mkdir(folder);
    end

    fn_affine = ['Affine_' filename(1:idx-1) '_seg1.mat'];
    fn_flow   = ['u_a_' filename(1:idx-1) '_seg1' filename(idx:end)];
    fn        = [sub '_' sess '_' model '_space-T1w_' param '_thr-Rsqmb,p=0.05B.nii.gz'];
    fn_path   = fullfile(folder, fn);
    gunzip(fn_path)
    % delete(fn_path)
    fn_param     = erase(fn_path, '.gz');
    fn_mask   = ['c_' filename(1:idx-1) '_pcereb' filename(idx:end)];

    job.subj.affineTr={fn_affine};
    job.subj.flowfield={fn_flow};
    job.subj.resample={fn_param};
    job.subj.mask={fn_mask};
    job.interp = 0; 

    suit_reslice_dartel(job)
end

% Flat map
for i = 1:length(subject_list)
    sub = subject_list{i};
    
    % mu in suit space
    fn_param_suit = ['wd' sub '_' sess '_' model '_space-T1w_' param '_thr-Rsqmb,p=0.05B.nii'];
    
    % go to the sub directory
    fullpath = fullfile(rowpath, sub, sess, model);
    cd(fullpath)

    % interpolation from vol to surf 
    map = suit_map2surf(fn_param_suit, 'stats', 'mode');

    if strcmp(param, 'mu')
        % Define key colors
        colors = [
            1 0 0;       % Red
            1 0.5 0;     % Orange
            1 1 0;       % Yellow
            0 1 0;       % Green
            0 1 1;       % Cyan
            0 0 1;       % Blue
            1 0 1        % Magenta
        ];
        
        % Number of steps in the final colormap
        n = 256;
        
        % Interpolate to get smooth transitions
        cmap = interp1( linspace(0,1,size(colors,1)), colors, linspace(0,1,n) );
        suit_plotflatmap(map, 'cmap', cmap, 'cscale', [1 9]);
        % Apply colormap to figure
        colormap(cmap);
        axis off     % removes x and y axes, ticks, and labels
        box off      % removes the border box around the plot
        
        % Add colorbar
        % Add colorbar at the bottom and make it smaller
        cb = colorbar('Location', 'southoutside');
        cb.Position(4) = cb.Position(4) * 0.6;  % adjust height
        caxis([1 9]); % Match the numerical range
    
    else
        cmap = hot(128);
        suit_plotflatmap(map, 'cmap', cmap, 'cscale', [0 0.5]);
        % Apply colormap to figure
        colormap(cmap);
        axis off     % removes x and y axes, ticks, and labels
        box off      % removes the border box around the plot
        
        % Add colorbar
        % Add colorbar at the bottom and make it smaller
        cb = colorbar('Location', 'southoutside');
        cb.Position(4) = cb.Position(4) * 0.6;  % adjust height
        caxis([0 0.5]); % Match the numerical range
    end

    % save figure
    filename = [fullpath '/' param '-flatmap.png'];
    exportgraphics(gcf, filename, 'Resolution', 300)
end

