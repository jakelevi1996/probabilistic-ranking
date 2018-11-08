clearvars

% Write names to CSV
load('tennis_data.mat')
fid = fopen('names.csv', 'w') ;
fprintf(fid, '%s\n', W{1:end}) ;
fclose(fid) ;

% Write games to CSV
csvwrite('games.csv', G-1)
