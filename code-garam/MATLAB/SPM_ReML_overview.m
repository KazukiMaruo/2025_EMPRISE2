% SPM ReML overiew
% _
% This script plots several variants for the variance components used
% in restricted maximum likelihood estimation (ReML) in SPM.
% 
% Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
% 2023-09-06, 12:30: first version


clear
close all

% specify covariances
n = 5;
p = 0.4;
TR= 2.1;
Q = cell(4,2);
Q{1,1} = eye(5);
Q{1,2} = toeplitz(p.^[0:1:(n-1)]);
Q{2,1} = eye(5);
Q{2,2} = toeplitz(p.^[0:1:(n-1)]) - eye(5);
C = spm_Ce('ar',n,p);
Q{3,1} = full(C{1});
Q{3,2} = full(C{2});
Q{4,1} = eye(5);
Q{4,2} = toeplitz([0, 1, zeros(1,n-2)]);
C = spm_Ce('fast',n,TR);

% plot covariances (ReML)
figure('Name', 'SPM ReML overview', 'Color', [1 1 1], 'Position', [50 50 1600 900]);
titles = {'What SPM says it''s doing', 'What some SPM slides are purporting', ...
          'What SPM actually does', 'Why don''t we just do this?'};

for i = 1:4
    for j = 1:2
        subplot(2,4,(i-1)*2+j);
        imagesc(Q{i,j});
        caxis([-0.1, +1.1]);
        axis square;
      % colormap jet;
        set(gca,'XTick',[0,10],'YTick',[0,10]);
        xlabel(sprintf('Q_{%d}', j), 'FontSize', 16);
        if j == 1, title(titles{i}, 'FontSize', 20); end;
        for k = 1:n
            text(k, 1, sprintf('%0.2f',Q{i,j}(1,k)), 'FontSize', 12, ...
                'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
        end;
        if i == 4 && j == 2
            text(6, 3, 'etc.', 'FontSize', 20, ...
                'HorizontalAlignment', 'Left', 'VerticalAlignment', 'Middle');
        end;
    end;
end;

% plot covariances (FAST)
figure('Name', 'SPM FAST overview', 'Color', [1 1 1], 'Position', [50 50 1600 900]);

for i = 1:numel(C)
    C{i} = full(C{i});
    subplot(3,6,i);
    imagesc(C{i});
    caxis([0, max(max(C{i}))]);
    axis square;
  % colormap jet;
    set(gca,'XTick',[0,10],'YTick',[0,10]);
    xlabel(sprintf('C_{%d}', i), 'FontSize', 16);
    if i == 2, title('What SPM FAST is doing', 'FontSize', 20); end;
    for k = 1:n
        text(k, 1, sprintf('%0.2f',C{i}(1,k)), 'FontSize', 10, ...
            'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
    end;
end;