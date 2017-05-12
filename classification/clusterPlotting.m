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
tic
[idx, C] = kmeans(data, k);
display([num2str(toc) ' s to do the clustering.'])

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
    %close(200 + sensor)
    clf
    figure(200 + sensor)
    [ha,pos] = tight_subplot(6, 1, 0, [0.1 0.031], [0.06 0.03]);
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
    pp = find(data(:,chan));   % find non-zero elements
    bott = min(data(pp,chan))   % find min of all non-zero elements
    bott = max(bott, 3);      % bottom of scale can't be less than ...
    ylim([0.9*bott 110*bott])
    %axis tight
    legg = legend(channels(chan,:), 'Location', 'NorthWest');
    set(legg,'Interpreter','none');
    set(legg,'FontSize',10);
    g = floor(log10(bott));
    set(gca, 'YTick', 10.^([g g+1 g+2]))
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
tic
pause(1.5)
set(gcf,'Position', [0 0 1900 1200])
hh = text(-1.3, 5000000, 'RMS Velocity [um/s]','Interpreter','Latex','rotation',90);
%% pretty print


%set(gcf,'Position', [400 0 700 1200])
set(gcf,'PaperPositionMode','auto')
            
fname = 'BLRMS_clusters';
rez = ['-r' num2str(300)];
%print('-depsc', rez, [fname '.eps'])
print('-dpng','-r100',['Figures/' fname '.png'])
%[a,~] = system(['makePDF.sh ' fname '.eps']);
%if a == 0
%    system(['rm ' fname '.eps']);
%end
toc

