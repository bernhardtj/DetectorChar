working_dir = '/Users/ramyad/Desktop/LIGO'
k = 3 % number of clusters
total_time = 5 % number of hours

data = load(strcat(working_dir,'/data_array.mat'));

% for some reason this array has a 13th column of zeros - need to fix in
% python script
data = data.data(:, 1:12);


% 5 cluster kmeans
[idx,C] = kmeans(data, k);

% generate time column for plotting 
time = linspace(0, total_time, length(idx));
data = [transpose(time) data];

% channel/frequency bands for plot labelling
mkdir(strcat(working_dir, '/clusterPlots'))
channels = {'0.03 to 0.1 Hz (X direction)'; '0.1 to 0.3 Hz (X direction)'; '0.3 to 1 Hz ( X direction)'; '1 to 3 Hz (X direction)'; 
    '3 to 10 Hz (X direction)'; '10 to 30 Hz (X direction)'; '0.03 to 0.1 Hz (Y direction)'; '0.1 to 0.3 Hz (Y direction)'; '0.3 to 1 Hz ( Y direction)'; '1 to 3 Hz (Y direction)'; 
    '3 to 10 Hz (Y direction)'; '10 to 30 Hz (Y direction)'};

% plot clusters for each channel and save
for channel = 2: size(data, 2) - 1
    for cluster = 1:k
        scatter(data(idx==cluster, 1), data(idx==cluster, channel), 2)
        hold on
    end
    ylim([0 inf])
    xlabel('Time (hours)')
    ylabel('RMS Velocity(micrometers/s)')
    title(strcat('Cluster Assignments for Channel', ' ', channels(channel, 1)))
    savefig(strcat(working_dir,'/clusterPlots/channel', num2str(channel - 1), 'clusters.fig'))
    hold off;

end
        
    
         
            
        
        
    

