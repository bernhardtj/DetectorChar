"""
util.py

Provides the following utilities:

- Logging
- INI Config files
- Progress Bars
- Data appending to HDF5 files
- User kwargs for common tasks
"""

from configparser import ConfigParser
from datetime import timedelta
from logging import Formatter, getLogger, FileHandler, StreamHandler
from os import makedirs
from os.path import abspath, join, exists

from gwpy.timeseries import TimeSeriesDict
from h5py import File

DEFAULT_LOG_LEVEL = 10
DEFAULT_LOG_PREFIX = '.'


def get_path(name, extension, prefix='.'):
    if not exists(prefix):
        makedirs(prefix)
    return join(abspath(prefix), f'{name}.{extension}')


def gen_formatter():
    return Formatter('%(asctime)s | %(name)s : %(levelname)s : %(message)s')


def get_logger(logname, log_level=DEFAULT_LOG_LEVEL, prefix=DEFAULT_LOG_PREFIX, verbose=False):
    logger = getLogger(logname)
    logger.setLevel(log_level)

    # set up FileHandler for output file
    log_path = get_path(logname, 'log', prefix=prefix)
    handlers = [FileHandler(log_path)]

    # set up handler for stdout
    if verbose:
        handlers.append(StreamHandler())

    # remove any existing handlers from logger to avoid repeated print statements
    for handler in logger.handlers[::-1]:
        # remove in reverse order. For some reason, not all handlers are removed unless we do this
        logger.removeHandler(handler)

    # add handlers to logger
    formatter = gen_formatter()
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger, log_path


def path2config(path):
    path = abspath(path)
    config = ConfigParser()
    config.read(path)
    return config, path


def config2bounds(bounds):
    ans = {}
    for bound in bounds.strip().split('\n'):
        key, min_val, max_val = bound.strip().split()
        ans[key] = (float(min_val), float(max_val))
    return ans


def config2dataclass(cls, cfg, channel):
    return [i(channel=channel,  # pass in channel
              **kwargunion(evalkwargs(**trycfg(cfg, i.__name__)),  # pass in dataclass-specific config section
                           evalkwargs(**trycfg(cfg, channel))))  # pass in channel-specific config section
            for i in cls.__subclasses__() if i.__name__ in eval(trycfg(cfg, channel)[cls.__name__])]


# list of evalkwargs imports:
timedelta


def evalkwargs(**kwargs):
    for key, val in kwargs.items():
        try:
            val = eval(val)  # NOTE, this may be fragile but should allow a lot of flexibility in how we specify options
        except Exception as e:  # NOTE: this could be fragile...
            pass
        kwargs[key] = val
    return kwargs


def kwargunion(*args):
    out = dict(args[0])
    for arg in args[1:]:
        out.update(dict(arg))
    return out


def trycfg(cfg, key):
    try:
        return cfg[key]
    except KeyError:
        return cfg['DEFAULT']


class Progress(object):
    def __init__(self, label, d: int, chars=60, quiet=False):
        self.label, self.d, self.chars, self.print = label, d, (chars - 7), self.noop if quiet else print
        self.print(f'{self.label}: [' + '-' * chars + f']', end='\r')

    def __enter__(self):
        d, chars = self.d, self.chars

        def progress(func, x: int, *args, **kwargs):  # io is a side effect B)
            self.print(f'{self.label}: [' + '=' * ((chars * x) // d) + f'({(100 * x) // d}%)=>', end='\r')
            return func(*args, **kwargs)

        return progress

    def __exit__(self, type, value, traceback):
        self.print(f'{self.label}: [' + '=' * self.chars + f'(DONE)=]')

    @staticmethod
    def noop(*args, **kwargs):
        pass


def path2h5file(path, mode='a'):
    return File(path, mode), path


# for a h5py dataset in a gwpy save file, we have dataset.attrs['x0'] == (data: TimeSeries).t0.seconds. However, it
# is useful to be able to quickly get (data: TimeSeries).times[-1].seconds without having to read out all the data.
# This is the hacky calculation that makes the appending support of hdf5 usable with gwpy.
get_last_time = lambda dataset: int(dataset.attrs['x0'] + (dataset.shape[0] - 1) * dataset.attrs['dx'])


def data_exists(channels, seg_stop: int, f: File):
    """Check for existing data up to a point, in the format created by write_to_disk."""

    for channel in channels:

        try:
            if get_last_time(f[channel]) < seg_stop:
                # this means the length of the existing data does not extend to where we intend to generate.
                break

        except KeyError:  # channel_file[channel] is not there.
            break

    else:
        # for all channels, it's likely this segment exists already on disk.
        return True

    return False


def write_to_disk(data_dict: TimeSeriesDict, seg_start: int, f: File):
    """Write a TimeSeriesDict to a gwpy-compatible .hdf5 file. Supports appending to an existing file."""

    for name in data_dict:

        # deal with each TimeSeries in the TimeSeriesDict.
        data = data_dict[name]

        try:

            # create a gwpy-compatible h5py file.
            data.write(f, **writing_opts)

        except RuntimeError:  # the RuntimeError in regard here is caused by the dataset already existing.

            # use the h5py File driver to get a direct pointer to the existing dataset.
            dataset = f[name]

            # compute the time offset between the existing data and the new data.
            secs = seg_start - get_last_time(dataset)
            padding = secs / dataset.attrs['dx']
            # print(f'write: padding from {get_last_time(dataset)} to {seg_start} ({secs}s, {padding}pts)')

            if data.value.shape[0] < -padding:

                # this would resize the dataset to be smaller than it already is.
                raise RuntimeError('insertion is not supported.')

            else:

                # append data to the end of the file.
                dataset.resize((dataset.shape[0] + padding + data.value.shape[0]), axis=0)
                dataset[-data.value.shape[0]:] = data.value
                f.flush()  # sync table to disk


# better fir window for anti-aliasing large TimeSeries decimation.
# see https://git.ligo.org/NoiseCancellation/GWcleaning/issues/2
better_aa_opts = lambda tser, fs: {'rate': fs, 'n': int(20 * tser.sample_rate.value / fs), 'window': 'blackmanharris'}

# always to TimeSeries.write(**writing_opts)
# these kwargs are passed down through gwpy and are ultimately evaluated by h5py create_dataset.
# this is done so that the TimeSeries dataset is created in an mode that supports appending.
writing_opts = {'compression': 'gzip', 'chunks': True, 'maxshape': (None,), 'append': True}
