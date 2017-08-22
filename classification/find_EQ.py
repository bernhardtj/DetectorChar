'''
Reads in earthquake channel data 
Timeshifts the data
Creates list of earthquake times based on peaks
Creates X and Y arrays for neural network
Saves arrays to mat file
'''

from __future__ import division

import numpy as np
import scipy.io as sio
from astropy.time import Time
import collections

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as sig

#plt.style.use('ggplot')
mpl.rcParams.update({
    'axes.grid': True,
    'axes.titlesize': 'medium',
    'font.family': 'serif',
    'font.size': 12,
    'grid.color': 'w',
    'grid.linestyle': '-',
    'grid.alpha': 0.5,
    'grid.linewidth': 1.5,
    'legend.borderpad': 0.2,
    'legend.fancybox': True,
    'legend.fontsize': 13,
    'legend.framealpha': 0.7,
    'legend.handletextpad': 0.1,
    'legend.labelspacing': 0.2,
    'legend.loc': 'best',
    'lines.linewidth': 1.5,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'text.usetex': False,
    'text.latex.preamble': r'\usepackage{txfonts}'
    })

mpl.rc("savefig", dpi=100)

# read in data
H1dat = sio.loadmat('Data/' + 'H1_SeismicBLRMS.mat')
edat  = np.loadtxt('Data/H1_earthquakes.txt')

#read in earthquake channels
cols   = [6,12,18,24,30,36,42,48]
vdat   = np.array(H1dat['data'][0])
vchans = np.array(H1dat['chans'][0])
for i in cols:
    add    = np.array(H1dat['data'][i])
    vdat   = np.vstack((vdat, add))
    vchans = np.append(vchans,H1dat['chans'][i])

# shift the data
t_shift = 0 # how many minutes to shift the data by
if t_shift > 0:
    for i in cols:
        add = np.array(H1dat['data'][i])
        for j in range(1, t_shift+1):        
            add_shift = add[j:]
            #print(np.shape(add_shift))
            add_values = np.zeros((j,1))
            add_shift = np.append(add_shift, add_values)
            #print(np.shape(add_shift))
            vdat = np.vstack((vdat, add_shift))
            chan = 'Time_Shift_' + str(j) + '_Min_EQ_Band_' + str(i)
            vchans = np.append(vchans, chan)
    vdat = vdat[:,:43200-t_shift]
size, points = np.shape(vdat)

print('The data has '+ str(size)+ ' columns and ' + str(points) + ' points')    


# convert UTC time to GPS time                      
times   = '2017-03-01 00:00:00'
t       = Time(times,format='iso',scale='utc')
t_start = int(np.floor(t.gps/60)*60)
dur_in_days = 30
dur_in_minutes = dur_in_days*24*60
dur     = dur_in_minutes * 60
t_end   = t_start + dur
t       = np.arange(t_start,t_end-t_shift, 60)
seconds_per_day = 24*60*60

#create list of earthquake times from text file 
#use peak ground motion to determine which earthquakes are bigger
row, col = np.shape(edat)
gdat = np.array([])
for i in range(row):
    point = edat[i][20]
    gdat = np.append(gdat,point)
gdat = gdat.T
glq = np.percentile(gdat,65)
#use only earthquakes with signifigant ground motion                          
row, col = np.shape(edat)
etime = np.array([])
for i in range(row):
    if (edat[i][20] >= glq):
        point = edat[i][5]
        etime = np.append(etime,point)
#use only earthqaukes that occur in March 2017         
col = len(etime)
etime_march = np.array([])
for i in range(col):
    if ((etime[i] >= t_start) and (etime[i] <= t_end)):
        point = etime[i]
        etime_march = np.append(etime_march,point)

# create list of earthquake times from peaks
# find peaks in all three z channel directions
widths  = np.arange(5, 140)   # range of widths in minutes
min_snr = 5
noise_perc = 15
peaks1 = sig.find_peaks_cwt(vdat[2], widths,
                                min_snr = min_snr, noise_perc=noise_perc)
peaks2 = sig.find_peaks_cwt(vdat[5], widths,
                                min_snr = min_snr, noise_perc=noise_perc)
peaks3 = sig.find_peaks_cwt(vdat[8], widths,
                                min_snr = min_snr, noise_perc=noise_perc)

# takes average time for earthquake times from three channels
# that are within dtau minutes of each other 
dtau = 3
peak_list = np.array([])
for i in peaks1:
    for j in peaks2:
        for k in peaks3:
            if (abs(i-j) <= dtau and abs(i-k) <= dtau):
                if __debug__:
                    print(str(i) + ' ' + str(j) + ' ' + str(k))
                avg = (i+j+k)/3
                peak_list = np.append(peak_list, avg)
EQ_locations = np.array([])
for i in peak_list:
    EQ_locations = np.append(EQ_locations, t[int(i)])

# assign X
X = vdat.T
if __debug__:
    print('Shape of the BLRMS data is ' + str(np.shape(X)))

# assign Y to 1 or 0 depending on whether there is an earthquake
Y = np.array([])   # this is the array of labels
eq_time = 5
for i in t:
    xlen = len(Y)
    for j in EQ_locations:
        if (j-eq_time <= i <= j+eq_time):
            Y = np.append(Y,1)
            break
    xlen2 = len(Y)
    if xlen == xlen2:
        Y  = np.append(Y,0)

# prints information about the shape of Y and the types of points in Y
print('Shape of Y (the EQ labels) is ' + str(np.shape(Y)))
print(collections.Counter(Y))

# saves data as mat file
# saves vdat and vchan for plotting 
data = {}
data['vdat']      = vdat
data['vchans']    = vchans
data['EQ_times']  = EQ_locations
data['X']         = X
data['EQ_labels'] = Y
data['t']         = t
sio.savemat('Data/EQ_info.mat', data,
                do_compression = True)

#Plot earthquakes determined by peaks and recorded earthquakes
'''
print("Producing plot...")
# Plot earthquakes determined by peaks
fig,axes  = plt.subplots(nrows=len(vdat), figsize=(40, 4*len(vdat)),
                             sharex=True)
for ax, data, chan in zip(axes, vdat, vchans):
    ax.scatter((t - t[0])/seconds_per_day, data ,
                   edgecolor='', c='purple', s=3,
                   label=r'$\mathrm{%s}$' % chan.replace('_','\_'),
                   rasterized=True)
    ax.set_yscale('log')
    ax.set_ylim(9, 1.1e4)
    ax.set_ylabel('RMS Velocity [nm/s]') 
    ax.set_xlabel('Time (after GPS ' + str(t[0]) + ') [days]')
    ax.grid(True, which='both')
    ax.legend()
    for e in range(len(EQ_locations)):
        ax.axvline(x=(EQ_locations[e] - t[0])/seconds_per_day,
                       color = 'blue', alpha=0.3, linewidth=3)
    for e in range(len(etime_march)):
        ax.axvline(x=(etime_march[e] - t[0])/seconds_per_day,
                       color = 'red', alpha=0.3, linewidth=3)

plt.xlim((0, (t[-1]-t[0])/seconds_per_day))
plt.title('Seismic BLRMS')
fig.tight_layout()
fig.savefig('Figures/EQ_peaks_indicated.pdf')

try:
    fig.savefig('/home/roxana.popescu/public_html/' + 'EQ_peaks_indicated.png')
except: 
    print(' ')
'''
