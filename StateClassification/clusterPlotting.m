%% this loads the seismic BLRMS data and clusters it using k-means
ifo = 'H1';
%working_dir = '/Users/ramyad/Desktop/LIGO'
k = 10;                  % number of clusters
total_time = 30 * 24;   % number of hours

%data = load(strcat(working_dir,'/data_array.mat'));
blrms = load(['Data/' ifo '_SeismicBLRMS_March.mat']);
zidx = 1:18;
data = blrms.data(zidx,:).';

% for some reason this array has a 13th column of zeros - need to fix in
% python script
%data = data.data(:, 1:12);

% k cluster kmeans
[idx, C] = kmeans(data, k);

%% generate time column for plotting 
% this code doesn't handle spans of missing data well
tt = linspace(0, total_time, length(idx)).';
%data = [time.' data];

% channel/frequency bands for plot labelling
%mkdir(strcat(working_dir, '/clusterPlots'))

channels = blrms.chans(zidx,:);

% plot clusters for each channel and save
c = distinguishable_colors(k);



for sensor = 2
    close(200 + sensor)
    figure(200 + sensor)
    [ha,pos] = tight_subplot(6, 1, 0, [0.1 0.031], [0.12 0.03]);
    for v = 1:6
      axes(ha(v));
      for clust = 1:k
        n = find(idx == clust);
        chan = 6*(sensor-1) + v;
        scatter(tt(n)/24, data(n, chan), 2, c(clust,:))
        hold on
      end
    set(gca,'YScale','log')
    set(gca,'XTick',[])
    bott = min(data(:,chan));
    bott = max(bott, 30);
    ylim([0.9*bott 100*bott])
    axis tight
    legg = legend(channels(chan,:), 'Location', 'NorthWest');
    set(legg,'Interpreter','none');
    set(legg,'FontSize',10)
    %title(strcat('Cluster Assignments for Channel', ' ', channels(channel, 1)))
    %savefig(strcat(working_dir,'/clusterPlots/channel', num2str(channel - 1), 'clusters.fig'))
    end
    set(gca,'Xtick',[0:7:42])
    xlabel('Time (days)')
    %ylabel('RMS Velocity (micrometers/s)')
    %legg = legend(channels, 'Location', 'SouthWest');
    %set(legg,'Interpreter','none');
    hold off
end
pause(3)
set(gcf,'Position', [400 0 700 1200])
        
%% pretty print


%set(gcf,'Position', [400 0 700 1200])
set(gcf,'PaperPositionMode','auto')
            
fname = 'BLRMS_clusters';
rez = ['-r' num2str(300)];
%print('-depsc', rez, [fname '.eps'])
print('-dpng','-r100',[fname '.png'])
%[a,~] = system(['makePDF.sh ' fname '.eps']);
%if a == 0
%    system(['rm ' fname '.eps']);
%end
        
    

