### GWpy
The getData.py script encounters issues for getting data over ~ 5 hours, resulting in the error 'Requested data could not be found'. After trying a number of ways of interacting with nds2, I could not resolve the issue. Nds2 also allows you to set parameters for your connection, including a GAP_HANDLER to deal with missing data. This looked quite promising, but in my code, I was not able to access this method on the connection object. 
I turned to GWPy for some help with this. GWPy has a TimeSeries module for accessing data from LIGO channels:
 
```
lho = TimeSeries.fetch(channel, 1174003218, 1174089618, 0,'nds40.ligo.caltech.edu', 31200, verbose=True)

```
However, the same error showed up when I tried to access 24-hour data (I guess it should be no surprise since this was an nds2 error). I found a similar issue posted on the gwpy Github:
```
When using the --nds option, if data are missing gw_summary will return a RuntimeError: Requested data were not found error and crash rather than padding with zeros or producing an empty plot.
```
TimeSeries.get has an argument called "pad" for a value to fill gaps in the data with, the default for which is 'don't fill gaps'. Right now, I'm trying to use this method with the pad argument. Using TimeSeries.get as is asks for LIGO access credentials however, but this method can take in keyword arguments to pass into TimeSeries.fetch, so this is currently what I'm trying:

```
first_channel = TimeSeries.fetch(channels[0], 1174003218, 1174006818, 0, host='nds40.ligo.caltech.edu', port=31200)
data_array = np.transpose(np.matrix(first_channel))
for channel in channels[1:]:
    lho = TimeSeries.get(channel, 1174003218, 1174089618, 0, host='nds40.ligo.caltech.edu', port=31200, verbose=True)
    data_array = np.hstack((data_array, np.transpose(np.matrix(lho))))
```

SIDE NOTE: GWpy install had a number of issues. Make sure whoever is trying to install it goes to the latest install docs https://gwpy.github.io/docs/latest/install.html because the dependencies have changed. I also had to add the package location to my Jupyter Notebook path: 
```
sys.path.append('/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')

```
After that, I got some sort of relative import error, but uninstalling and reinstalling gwpy through pip seems to have solved it for now. 
