#!/usr/bin/env python
# # Get Minute Trend Data from the LIGO Sites


# Library Imports and Python parameter settings
from __future__ import division
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as sio
from timeit import default_timer as timer
#import scipy.signal as sig
#import scipy.constants as const
from astropy.time import Time
import sys
#sys.path.append('/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7')
import nds2

# input argument Parsing
if isinstance(sys.argv[1], str):
    ifo = sys.argv[1]
else:
    ifo = 'L1'

# ## setup the servers, start times, and duration
# Setup connection to the NDS
if   ifo == 'H1':
    ndsServer  = 'nds.ligo-wa.caltech.edu'
elif ifo == 'L1':
    ndsServer  = 'nds.ligo.caltech.edu'
elif ifo == 'C1':
    ndsServer  = 'nds40.ligo.caltech.edu'


portNumber = 31200
conn       = nds2.connection(ndsServer, portNumber)

# Setup start and stop times
times   = '2017-03-01 00:00:00'
t       = Time(times, format='iso', scale='utc')
#t_start = int(t.gps)
# round start time to multiple of 60 for minute trend
t_start = int(np.floor(t.gps/60)*60)

dur_in_days    = 30
dur_in_minutes = dur_in_days * 24 * 60
dur            = dur_in_minutes * 60    # must be a multiple of 60


# ## Build up the channel list and Get the Data
chan_head = ifo + ':' + 'ISI-' + 'GND_STS' + '_'
sensors   = ['ETMX', 'ETMY', 'ITMY']
dofs      = ['X', 'Y', 'Z']
bands     = ['30M_100M', '100M_300M', '300M_1', '1_3', '3_10', '10_30']
channels  = []
# why is the channel ordering so weird? 
# need to use sorted to preserve the intended ordering
for sensor in sensors:
    for dof in dofs:
        for band in bands:
            channel = chan_head + sensor + '_' + dof + '_BLRMS_' + band + '.mean, m-trend'
            channels.append(channel)

print("Getting data from " + ndsServer + "...")
tic  = timer()
data = conn.fetch(t_start, t_start + dur, channels)
toc  = timer()
print(str(round(toc - tic, 2)) + " seconds elapsed.")

if __debug__:
    for i in channels:
        print(i)

# save the data so that it can be loaded by matlab or python
# savemat will compress the data and save it in hdf5 format
vdata = []
# get the data and stack it into a single matrix
# where the data are the columns
for k in range(len(channels)):
    vdata.append(data[k].data)


# save to a hdf5 format that matlab can read
# (why is compression off by default?)
funame = 'Data/' + ifo + '_SeismicBLRMS.mat'
sio.savemat(funame, mdict={'data': vdata, 'chans': channels, 't_start': t_start},
                do_compression=True)
print("Data saved as " + funame)

# ### some debugging info about the channels
if __debug__:
    print("Channel name is "      + data[0].channel.name)
    print("Sample rate is "       + str(data[0].channel.sample_rate) + " Hz")
    print("Number of samples is " + str(data[0].length))
    print("GPS Start time is "    + str(data[0].gps_seconds))
