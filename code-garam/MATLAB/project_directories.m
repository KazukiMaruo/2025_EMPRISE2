% EMPRISE: project directories
% _
% Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
% 2023-05-17, 10:33: first version
% 2023-05-25, 11:11: renamed "work_dir" to "prep_dir"
% 2023-07-04, 14:43: adapted tools directory


% create project directories
stud_dir = 'C:\Joram\projects\MPI\EMPRISE\';
data_dir = strcat(stud_dir,'data\');
prep_dir = strcat(data_dir,'derivatives\fmriprep\');
stat_dir = strcat(stud_dir,'stats\');
tool_dir = 'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\EMPRISE\code\MATLAB\';

% save project directories
save('project_directories.mat', 'stud_dir', 'data_dir', 'prep_dir', 'stat_dir', 'tool_dir');