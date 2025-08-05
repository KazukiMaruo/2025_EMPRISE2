function check_func_imgs(func_imgs)
% _
% Checks functional image files before or after preprocessing
% FORMAT check_func_imgs(func_imgs)
% 
%     func_imgs - an R x 1 cell array where each entry is the 
%                 filepath to a 4D NIfTI filepath from one run
% 
% FORMAT check_func_imgs(func_imgs) loads the images specified by func_imgs
% and plots their entire correlation matrix, run-wise mean correlations and
% k-means clustering results.
% 
% Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
% 2023-05-22, 12:44: first version


%%% Step 1: load images %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load images
num_vol = 0;
run     = [1:numel(func_imgs)];
x_ticks = zeros(1,numel(run));
for i = 1:numel(run)
    H = spm_vol(func_imgs{i});
    Y = spm_read_vols(H);
    S(i).row   = [(num_vol+1):(num_vol+numel(H))];
    S(i).Y     = reshape(Y,[prod(H(1).dim), numel(H)])';
    x_ticks(i) = num_vol + 1;
    num_vol    = num_vol + numel(H);
end;
clear H Y


%%% Step 2: compute correlations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% calculate correlations
Y = vertcat(S.Y);
R = corr(Y');

% average correlations
r_in  = zeros(1,numel(run));
r_out = zeros(1,numel(run));
for i = 1:numel(run)
    r_in(i)  = mean(mean(R(S(i).row,S(i).row)));
    r_out(i) = mean(mean(R(horzcat(S(run~=i).row),S(i).row)));
end;

% perform k-means
num_cls = [2,3,4];
idx     = zeros(numel(num_cls),num_vol);
for i = 1:numel(num_cls)
    idx(i,:) = kmeans(Y,num_cls(i))';
end;


%%% Step 3: visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% visualize correlations
figure('Name', 'mcheck: check functional images', 'Color', [1 1 1], 'Position', [50 50 1200 800]);

subplot(2,2,1); hold on;
cmap = colormap('jet');
cmap = flipud(cmap);
colormap(cmap);
imagesc(R);
caxis([0.75, 1]);
axis([(1-0.5), (num_vol+0.5), (1-0.5), (num_vol+0.5)]);
axis ij square;
colorbar;
set(gca,'XTick',x_ticks,'XTickLabels',cellstr(num2str(run'))');
set(gca,'YTick',x_ticks,'YTickLabels',cellstr(num2str(run'))');
xlabel('runs, volumes', 'FontSize', 12);
ylabel('runs, volumes', 'FontSize', 12);
title('all correlations', 'FontSize', 16);

subplot(2,2,2); hold on;
plot(run, r_in,  '-og', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0,176,80]./255,  'MarkerFaceColor', [0,176,80]./255);
plot(run, r_out, '-ob', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0,176,240]./255, 'MarkerFaceColor', [0,176,240]./255);
axis([(1-0.5), (numel(run)+0.5), 0.75, 1]);
set(gca,'Box','On');
set(gca,'XTick',run);
legend({'within runs', 'between runs'}, 'Location', 'SouthEast');
xlabel('run', 'FontSize', 12);
ylabel('correlation', 'FontSize', 12);
title('mean correlations', 'FontSize', 16);

subplot(7,1,5);
% cmap = [248, 251, 13; 169, 251, 87; 254, 183, 0]./255;
% colormap(cmap);
imagesc(idx(2,:));
axis([(1-0.5), (num_vol+0.5), 0.5, 1.5]);
set(gca,'XTick',x_ticks,'XTickLabels',cellstr(num2str(run'))');
set(gca,'YTick',[0, 2]);
ylabel('k = 3', 'FontSize', 12);

subplot(7,1,7);
% cmap = [248, 251, 13; 254, 183, 0]./255;
% colormap(cmap);
imagesc(idx(1,:));
axis([(1-0.5), (num_vol+0.5), 0.5, 1.5]);
set(gca,'XTick',x_ticks,'XTickLabels',cellstr(num2str(run'))');
set(gca,'YTick',[0, 2]);
xlabel('runs, volumes', 'FontSize', 12);
ylabel('k = 2', 'FontSize', 12);