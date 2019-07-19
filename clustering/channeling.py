"""
channeling.py
Makes a post-processed channel

Provides:
process_channel()
implementations of PostProcessor

"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

from gwpy.segments import DataQualityFlag
from gwpy.time import to_gps, from_gps
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from h5py import File
from numpy import median, zeros_like, arange, where, logical_and, setdiff1d, sqrt, NaN, isnan, interp, sum
from scipy.signal import spectrogram

from util import get_logger, path2config, get_path

# initialize logging.
logger, log_path = get_logger(__name__, verbose=True)
print(f'Writing log to: {log_path}')

config, config_path = path2config(get_path('config', 'ini'))
logger.info(f'Loaded config from: {config_path}')


@dataclass
class PostProcessor:
    """
    A dataclass interface to hold processing options and the processing computation function.
    There are two methods to be implemented.
    """
    channel: str  # internal channel name.

    # general config options.
    channels: List[str]  # channels to generate.
    stride_length: timedelta  # length of the strides.
    respect_segments: bool  # whether or not to overwrite unlocked time with zeros.

    # channel-specific config options
    postprocessor: list  # need to specify post-processors to instantiate on a per-channel basis.
    bands: List[Tuple[int]]  # BLRMS bands of form [(start1, stop1), (start2, stop2) ... (startN, stopN)] in Hz.

    # dependent attributes.
    @property
    def extra_seconds(self) -> int:
        return 0

    # required methods.
    @property
    def output_channels(self) -> List[str]:
        """Return a list of the possible channels to output for the PostProcessor implementation."""
        return NotImplemented

    def compute(self, raw: TimeSeries) -> TimeSeriesDict:
        """Computation function. Somehow processes a TimeSeries, producing one or more TimeSeries as a result."""
        return NotImplemented


@dataclass
class BLRMS(PostProcessor):
    """BLRMS álá Gabriele Vajente. Also the reference implementation of PostProcessor."""

    # BLRMS config options
    fs: int  # (down) re-sampling frequency [Hz]
    tn: int  # number of seconds for each FFT [s]
    to: float  # FFT overlap in seconds [s]
    df: int  # frequency width of running median for line identification [Hz]
    thr: int  # threshold for line identification
    dt: int  # window size for median smoothing and glitch removal [s]

    @property
    def extra_seconds(self) -> int:
        return int(self.tn + self.to)  # NOTE: a guess; I don't understand this well. Watch padding if this != 8.

    @property
    def output_channels(self) -> List[str]:
        variants = ['', '_nolines', '_noglitch', '_smooth', '_nolines_noglitch', '_nolines_smooth']
        return [f'{self.channel}_{self.__class__.__name__}_{band_start}_{band_stop}{variant}'
                for band_start, band_stop in self.bands for variant in variants]

    def compute(self, raw: TimeSeries) -> TimeSeriesDict:
        debug = f'compute_blrms ({self.channel}) : '

        # resample to specified frequency.
        # x = decimate(raw.value, int(raw.sample_rate.value / fs))
        raw = raw.resample(self.fs)

        # compute spectogram. Set up kwargs for creation of output TimeSeries.
        F, T, Sh = spectrogram(raw.value, nperseg=self.fs * self.tn, noverlap=int(self.fs * self.to), fs=self.fs)
        ts_kwargs = dict(t0=raw.t0, dt=T[1] - T[0], unit=raw.unit)
        logger.debug(debug + 'Computed scipy.spectrogram.')

        # identify lines by comparing the PSD to a calculated background.
        Sh_m = median(Sh, axis=1)  # get median PSD for each time.
        Nf = int(self.df / F[1])  # convert the median window in number of bins.
        Sb_m = zeros_like(Sh_m)  # start with empty background vector.
        # compute a windowed median of the PSD.
        for i, f in enumerate(F):
            # select the bins in the current window.
            idx = arange(max(i - Nf, 0), min(i + Nf, F.shape[0]))
            # compute estimate of background (without lines) by computing the median in the current window.
            Sb_m[i] = median(Sh_m[idx])
        else:
            logger.debug(debug + 'Estimated PSD of background.')
        # find all lines, i.e. all bins where the PSD is larger than thr times the background.
        line_idx = where(logical_and(F > 10, Sh_m / Sb_m > self.thr))[0]
        logger.debug(debug + 'Located the line frequencies.')

        # Compute BLRMS for all bands
        out = TimeSeriesDict()
        for band_start, band_stop in self.bands:
            channel_prefix = f'{self.channel}_BLRMS_{band_start}_{band_stop}'
            # select frequency bins.
            idx = arange(int(band_start / F[1]), int(band_stop / F[1]))
            # full BLRMS, using all bins.
            out[channel_prefix] = TimeSeries(sum(Sh[idx, :], axis=0), **ts_kwargs)
            # remove the index of all lines.
            idx = setdiff1d(idx, line_idx)
            # compute BLRMS excluding lines.
            out[f'{channel_prefix}_nolines'] = TimeSeries(sum(Sh[idx, :], axis=0), **ts_kwargs)

            # Time-domain median smoothing and glitch removal
            for prefix in channel_prefix, f'{channel_prefix}_nolines':
                blrms = out[prefix].value
                # convert the time-domain median window size from seconds to BLRMS samples
                NT = int(self.dt / T[1])
                # empty vectors for running median and running 'median-stdev'
                blrms_m = zeros_like(blrms)
                blrms_rms = zeros_like(blrms)
                # loop over all time samples
                for i, x in enumerate(blrms):
                    # select samples in current window
                    idx = arange(max(i - NT, 0), min(i + NT, T.shape[0]))
                    # median of the BLRMS in the current window
                    blrms_m[i] = median(blrms[idx])
                    # median-based equivalent of the variance
                    blrms_rms[i] = median((blrms[idx] - blrms_m[i]) ** 2)
                # identify all glitch times as samples when the BLRMS deviated from median more than 3 times the median-stdev
                glitch_idx = where((blrms - blrms_m) > 3 * sqrt(blrms_rms))[0]
                # remove the glitchy times
                blrms_noglitch = blrms.copy()  # first set glitchy times to NaN
                blrms_noglitch[glitch_idx] = NaN
                idx = isnan(blrms_noglitch)  # then find the samples that are glitchy
                # linear interpolation using values around glitches
                blrms_noglitch[idx] = interp(T[idx], T[~idx], blrms[~idx])
                # save results to dictionary
                out[f'{prefix}_smooth'] = TimeSeries(blrms_m, **ts_kwargs)
                out[f'{prefix}_noglitch'] = TimeSeries(blrms_noglitch, **ts_kwargs)

        # F_lines = F[line_idx]
        # lines = {'F_lines': F_lines, 'F': F, 'Smedian': Sh_m, 'Sbg': Sb_m, 'line_idx': line_idx}

        # fix channel names.
        for i in out:
            out[i].name = i

        return out


def process_channel(processor: PostProcessor, start: datetime, stop: datetime) -> str:
    """
    Post-processes a channel using the given post-processor, and streams to a file in the working directory.
    The output .hdf5 file is given by the channel name and the start time.
    This is because inserting (unsupported) requires reading out the whole database and re-writing it again.
    It's not a terribly high priority, I think.

    :return filename of generated post-processed channel.

    >>> from channeling import config, process_channel, PostProcessor
    >>> from util import config2dataclass
    >>>
    >>> for channel in eval(config['DEFAULT']['channels']):
    >>>     for processor in config2dataclass(PostProcessor, config, channel):
    >>>         process_channel(processor, start, stop)

    or even
    >>> from channeling import config, process_channel, PostProcessor
    >>> from util import config2dataclass
    >>> from multiprocessing import Pool
    >>>
    >>> p = lambda channel: [process_channel(processor, start, stop) for processor in
    >>>      config2dataclass(PostProcessor, config, channel)]
    >>>
    >>> pool = Pool()
    >>> pool.map(p, eval(config['DEFAULT']['channels']))

    """

    # for a h5py dataset in a gwpy save file, we have dataset.attrs['x0'] == (data: TimeSeries).t0.seconds. However, it
    # is useful to be able to quickly get (data: TimeSeries).times[-1].seconds without having to read out all the data.
    # This is the hacky calculation that makes the appending support of hdf5 usable with gwpy.
    get_last_time = lambda dataset: int(dataset.attrs['x0'] + (dataset.shape[0] - 1) * dataset.attrs['dx'])

    def write_to_disk(data_dict: TimeSeriesDict, seg_start: int, f: File):
        """Write a TimeSeriesDict to a gwpy-compatible .hdf5 file. Supports appending to an existing file."""

        for name in data_dict:

            # deal with each TimeSeries in the TimeSeriesDict.
            data = data_dict[name]

            try:

                # create a gwpy-compatible h5py file.
                # these kwargs are passed down through gwpy and are ultimately evaluated by h5py create_dataset.
                # this is done so that the TimeSeries dataset is created in an mode that supports appending.
                data.write(f, compression="gzip", chunks=True, maxshape=(None,))

            except RuntimeError:  # the RuntimeError in regard here is caused by the dataset already existing.

                # use the h5py File driver to get a direct pointer to the existing dataset.
                dataset = f[name]

                # compute the time offset between the existing data and the new data.
                last_time = get_last_time(dataset)
                secs = seg_start - last_time
                padding = secs / dataset.attrs['dx']
                logger.debug(f'write: padding from {last_time} to {seg_start} ({secs}s, {padding}pts)')

                if data.value.shape[0] < -padding:

                    # this would resize the dataset to be smaller than it already is.
                    raise RuntimeError('insertion is not supported.')

                else:

                    # append data to the end of the file.
                    dataset.resize((dataset.shape[0] + padding + data.value.shape[0]), axis=0)
                    dataset[-data.value.shape[0]:] = data.value
                    f.flush()  # sync table to disk

            logger.info(f'Wrote {name} to {f}')

    # use h5py to make a mutable object pointing to a file on disk.
    filename = get_path(f'{processor.channel} {start}', 'hdf5')
    channel_file = File(filename, 'a')
    logger.debug(f'Initiated hdf5 stream to {filename}')

    # get the number of strides.
    num_strides = (stop - start) // processor.stride_length

    # create list of start and end times.
    strides = [[start + processor.stride_length * i,
                start + processor.stride_length * (i + 1)]
               for i in range(num_strides)]

    # stride loop.
    for stride_start, stride_stop in strides:

        # check for existing data.
        for channel in processor.output_channels:  # get expected channel names from the processor.

            try:
                if from_gps(get_last_time(channel_file[channel])) < stride_stop:
                    # this means the length of the existing data does not extend to where we intend to generate.
                    break

            except KeyError:  # channel_file[channel] is not there.
                break

        else:
            # for all possible output channels, it's likely this stride exists already on disk.
            continue

        # get the data.
        logger.debug(f'Initiating data download for {processor.channel} ({stride_start} to {stride_stop})')

        # separately download all observing segments within the stride, or one segment for the whole stride.
        # this is set by the processor.respect_segments: bool option.
        raw_segments = [[TimeSeries.get(processor.channel, seg_start, seg_stop + processor.extra_seconds), seg_start]
                        for i, seg_start, seg_stop in (
                            [[i, int(s.start), int(s.end)] for i, s in enumerate(DataQualityFlag.query(
                                'L1:DMT-ANALYSIS_READY:1', to_gps(stride_start), to_gps(stride_stop)).active)]
                            if processor.respect_segments else
                            [[0, to_gps(stride_start).seconds, to_gps(stride_stop).seconds]])]

        logger.info(f'Completed data download for {processor.channel} ({stride_start} to {stride_stop})')

        for raw, segment_start in raw_segments:
            # use the processor to compute each downloaded segment in the stride.
            finished_segment = processor.compute(raw)
            logger.info(f'Generated {processor.__class__.__name__} for {processor.channel}')

            # write each computed segment to the channel file.
            write_to_disk(finished_segment, segment_start, channel_file)

        logger.info(f'Completed stride {stride_start} to {stride_stop})')

    logger.debug(f'Completed channel at {filename}')

    # for automated usage of the post-processed data, return the generated filename.
    return filename
