function [raw, hdr] = tsvload(filename)
% _
% Load TSV-file with non-numeric data
% FORMAT [raw, hdr] = tsvload(filename)
% 
%     filename - a string, the filepath to the TSV file
% 
%     raw      - an n x c cell array of file contents
%     hdr      - a  1 x c cell array of file header names
%                o  n = number of rows in the TSV file (excl. header)
%                o  c = number of columns in the TSV file
% 
% FORMAT [raw, hdr] = tsvload(filename) loads the file specified by
% filename and returns its contents as cell array raw and its header as
% cell array hdr. If all values in one column can be converted to numbers,
% then they will all be numeric, otherwise they will all be strings.
% 
% Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
% 2023-05-17, 08:58: first version


% load file contents
f = fopen(filename);
lines = textscan(f,'%s','Delimiter','\n');
lines = lines{1};

% unpack header
cp  = strfind(lines{1},char(9));
cp  = [0, cp, numel(lines{1})+1];
col = numel(cp)-1;
hdr = cell(1,col);
for j = 1:col
    % store current entry from header
    hdr{j} = lines{1}((cp(j)+1):(cp(j+1)-1));
end;

% unpack lines
row  = numel(lines)-1;
raw  = cell(row,col);
conv = false(row,col);
for i = 1:row
    line = lines{1+i};
    cp   = strfind(line,char(9));
    cp   = [0, cp, numel(line)+1];
    for j = 1:col
        % store current entry from data set
        raw{i,j} = line((cp(j)+1):(cp(j+1)-1));
        % if convertible to numeric, take a note
        if ~isempty(str2num(raw{i,j}))
            conv(i,j) = true;
        end;
    end;
end;

% convert to numerical
conv = all(conv);
for j = 1:col
    % convert column, if all rows allow converting
    if conv(j)
        for i = 1:row
            raw{i,j} = str2num(raw{i,j});
        end;
    end;
end;