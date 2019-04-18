%% this loads the seismic BLRMS data and clusters it using k-means
ifo = 'H1';

k = 12;                 % number of clusters
total_time = 30 * 24;   % number of hours

% this is minute-mean trend of the seismometers
% so each column is data and there is no time vector
blrms = load(['Data/' ifo '_SeismicBLRMS.mat']);
[a,b] = size(blrms.data);
zidx = 1:a;
data = blrms.data(zidx,:).';



% k cluster kmeans
opts = statset('Display','iter', 'UseSubstreams', 1);
tic
vox = log10(data+1);         % take the log; it seems better
vox = data;
[idx, C] = kmeans(vox, k,...
                   'Replicates', 5,...
                   'Distance','sqeuclidean');
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

n_bands = 6;   % number of frequency bands per channel

for sensor = 9
    %close(200 + sensor)
    clf
    figure(200)
    [ha,pos] = tight_subplot(6, 1, 0, [0.06 0.031], [0.06 0.03]);
    for v = 1:n_bands     % go through all the frequency bands for this sensor DoF
      axes(ha(v));
      chan = n_bands*(sensor-1) + v;
      for clust = 1:k
        n = find(idx == clust);
        scatter(tt(n)/24, data(n, chan), 2, c(clust,:))
        hold on
      end
      set(gca,'YScale','log')
      set(gca,'XTick',[])
      
      pp   = find(data(:,chan));   % find non-zero elements
      bott = min(data(pp,chan));   % find min of all non-zero elements
      bott = max(bott, 3);      % bottom of scale can't be less than ...
      ylim([0.9*bott 110*bott])
      %axis tight
      
      chan_str = channels(chan,:);
      rr = strfind(chan_str, 'BLRMS');
      rs = strfind(chan_str, '.mean');
      if v==1
          title_str = chan_str(1:(rr-2));
          title(title_str,...
                'Interpreter', 'none')
      end
      
      legg = legend(chan_str((rr+6):(rs-1)),...
                    'Location', 'NorthWest');
      set(legg, 'Interpreter', 'none');
      set(legg, 'FontSize'   , 10);
      
      g = floor(log10(bott));
      set(gca, 'YTick', 10.^([g g+1 g+2]));
      
    end
    set(gca, 'Xtick', [0:1:32])
    xlabel('Time [days]',...
           'Interpreter','Latex')
    %ylabel('RMS Velocity (micrometers/s)')
    %legg = legend(channels, 'Location', 'SouthWest');
    %set(legg,'Interpreter','none');
    hold off
end
tic
pause(1.5)
set(gcf,'Position', [0 0 1420 790])
hh = text(-1.3, 5000000, 'RMS Velocity [$\mu$m/s]',...
          'Interpreter','Latex',...
          'rotation',90);
%% pretty print


%set(gcf,'Position', [400 0 700 1200])
set(gcf,'PaperPositionMode','auto')
            
fname = 'BLRMS_clusters';
rez = ['-r' num2str(300)];
%print('-depsc', rez, [fname '.eps'])
%print('-dpng','-r100',['Figures/' fname '.png'])
%[a,~] = system(['makePDF.sh ' fname '.eps']);
%if a == 0
%    system(['rm ' fname '.eps']);
%end
toc

