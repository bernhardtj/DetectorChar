#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy.io as sio
from timeit import default_timer as time
from astropy.time import Time
from gwpy.timeseries import TimeSeries


# Setup start and stop times
times   = '2017-02-01 00:00:00'
t       = Time(times, format='iso', scale='utc')
t_start = int(np.floor(t.gps/60)*60)

#dur_in_days    = (28+31+30+31+30)
dur_in_days = 15
dur_in_minutes = dur_in_days * 24 * 60
dur            = dur_in_minutes * 60    # must be a multiple of 60

ifo = 'H1'
#channel = 'PEM-EY_WIND_ROOF_WEATHER_MPS.mean,m-trend'
#print('Fetching data')
#data = TimeSeries.fetch(ifo + ':' + channel, t_start, t_start+dur)
#print('Fetched data')

#funame = ifo + channel + '.mat'
#sio.savemat(funame, mdict={'data': vdata, 't_start': t_start},
#                do_compression=True)
#print("Data saved as " + funame)

chan_head = ifo + ':' + 'ISI-' + 'GND_STS' + '_'
sensors   = ['ETMX', 'ETMY', 'ITMY']
dofs      = ['X', 'Y', 'Z']
bands     = ['30M_100M', '100M_300M', '300M_1', '1_3', '3_10', '10_30']
channels  = []
for sensor in sensors:
    for dof in dofs:
        for band in bands:
            channel = chan_head + sensor + '_' + dof + '_BLRMS_' + band + '.mean, m-trend'
            channels.append(channel)

data = np.array([])
data = TimeSeries.fetch(channels[0], t_start,t_start+dur)
for i in channels[1:]:
    print('Fetching ' + i)
    add = TimeSeries.fetch(i, t_start, t_start+dur)
    data = np.vstack((data, add))
print('Fetched ' + i)

funame = ifo + '_Months_data.mat'
sio.savemat(funame, mdict={'data': vdata, 't_start': t_start},
                do_compression=True)
