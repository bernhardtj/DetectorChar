"""
channeling.py
Makes a post-processed channel

Provides:
process_channel()
channeling_reader()
implementations of PostProcessor

"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

from gwpy.segments import DataQualityFlag
from gwpy.time import to_gps
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from numpy import median, zeros_like, arange, where, logical_and, setdiff1d, sqrt, NaN, isnan, interp, sum, unique
from scipy.signal import spectrogram

from util import get_logger, path2config, get_path, write_to_disk, better_aa_opts, data_exists, config2dataclass, \
    path2h5file

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
    bands: List[Tuple[int, int]]  # BLRMS bands of form [(start1, stop1), (start2, stop2) ... (startN, stopN)] in Hz.

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
        raw = raw.resample(**better_aa_opts(raw, self.fs))

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


@dataclass
class RawCache(PostProcessor):
    """Downloads and saves the data."""

    @property
    def output_channels(self) -> List[str]:
        # for some reason the commas are not being escaped.
        # chop off the last bit for saved minute-trend data.
        if self.channel.endswith(',m-trend'):
            return [self.channel.replace(',m-trend', '')]
        else:
            return [self.channel]

    def compute(self, raw: TimeSeries) -> TimeSeriesDict:
        out = TimeSeriesDict()
        if self.channel.endswith(',m-trend'):
            raw.name = raw.name.replace(',m-trend', '')
        out[raw.name] = raw
        return out


@dataclass
class MinuteTrend(PostProcessor):
    """Makes minute-trend data."""

    @property
    def output_channels(self) -> List[str]:
        return [self.channel + '.mean']

    def compute(self, raw: TimeSeries) -> TimeSeriesDict:
        out = TimeSeriesDict()
        times = unique([60 * (t.value // 60) for t in raw.times])
        raw.name = raw.name + '.mean'
        out[raw.name] = TimeSeries([raw.crop(t - 60, t).mean().value for t in times[1:]], times=times[:-1])
        out[raw.name].__metadata_finalize__(raw)
        return out


@dataclass
class NormMinuteTrend(PostProcessor):
    """Makes minute-trend data normalized by their standard score."""
    # NOTE: it just works if you use a single stride. 

    @property
    def output_channels(self) -> List[str]:
        return [self.channel + '.mean_norm']

    def compute(self, raw: TimeSeries) -> TimeSeriesDict:
        out = TimeSeriesDict()
        times = unique([60 * (t.value // 60) for t in raw.times])
        raw.name = raw.name + '.mean_norm'
        mean = raw.mean().value
        std = raw.std().value
        out[raw.name] = TimeSeries([(raw.crop(t - 60, t).mean().value - mean) / std for t in times[1:]], times=times[:-1])
        out[raw.name].__metadata_finalize__(raw)
        return out


def process_channel(processor: PostProcessor, start: datetime, stop: datetime, downloader=TimeSeriesDict.get) -> str:
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

    # use h5py to make a mutable object pointing to a file on disk.
    channel_file, filename = path2h5file(get_path(f'{processor.channel} {start}', 'hdf5'))
    logger.debug(f'Initiated hdf5 stream to {filename}')

    # get the number of strides.
    num_strides = (stop - start) // processor.stride_length

    # create list of start and end times.
    strides = [[start + processor.stride_length * i,
                start + processor.stride_length * (i + 1)]
               for i in range(num_strides)]

    # stride loop.
    for stride_start, stride_stop in strides:

        if data_exists(processor.output_channels, to_gps(stride_stop).seconds, channel_file):
            # for all possible output channels, it's likely this stride exists already on disk.
            continue

        # get the data.
        logger.debug(f'Initiating data download for {processor.channel} ({stride_start} to {stride_stop})')

        # separately download all observing segments within the stride, or one segment for the whole stride.
        # this is set by the processor.respect_segments: bool option.
        # it really should be processor.respect_segments: str = 'L1:DMT-ANALYSIS_READY:1' for generality.
        segments = [[int(s.start), int(s.end)] for s in
                    DataQualityFlag.query('L1:DMT-ANALYSIS_READY:1', to_gps(stride_start), to_gps(stride_stop)).active
                    ] if processor.respect_segments else [[to_gps(stride_start).seconds, to_gps(stride_stop).seconds]]

        raw_segments = list()
        for seg_start, seg_stop in segments:
            try:
                raw_segments.append([downloader([processor.channel],
                                                start=seg_start, end=seg_stop + processor.extra_seconds), seg_start])
            except RuntimeError:  # sometimes the data does not exist on the server. The show must go on, though.
                logger.warning(f'SKIPPING download for {processor.channel} ({stride_start} to {stride_stop}) !!')

        logger.info(f'Completed data download for {processor.channel} ({stride_start} to {stride_stop})')

        for raw, segment_start in raw_segments:
            # use the processor to compute each downloaded segment in the stride.
            finished_segment = processor.compute(raw[processor.channel])
            logger.info(f'Generated {processor.__class__.__name__} for {processor.channel}')

            # write each computed segment to the channel file.
            write_to_disk(finished_segment, segment_start, channel_file)

        logger.info(f'Completed stride {stride_start} to {stride_stop})')

    logger.debug(f'Completed channel at {filename}')

    # for automated usage of the post-processed data, return the generated filename.
    return filename


def channeling_reader(in_channels: List[str], generation_start: datetime, search_dirs: List[str] = ('.',)):
    """
    Return an equivalent function to TimeSeriesDict.get() for retrieving post-processed data.
    The naming scheme for save files used by this script is f'{processor.channel} {start}.hdf5',
    in order to avoid insertion caused by starting at different times. As you probably already
    know from reading util.py, insertion is not supported.

    :param in_channels: list of names of input channels processed to create the searchable channels.
    :param generation_start: time of generation start for the input channels.
    :param search_dirs: places to search for channeling files.
    :return: equivalent function to TimeSeriesDict.get() set to search with the parameters given.
    """

    def channeling_read(out_channels: List[str], **kwargs) -> TimeSeriesDict:
        out = TimeSeriesDict()

        for channel in out_channels:
            for prefix in search_dirs:
                for in_channel in in_channels:
                    try:
                        # lock the target file
                        h5file, _ = path2h5file(
                            get_path(f'{in_channel} {generation_start}', 'hdf5', prefix=prefix),
                            mode='r')
                        # read off the dataset
                        out[channel] = TimeSeries.read(h5file, channel, **kwargs)
                    except (FileNotFoundError, KeyError, OSError):
                        # file not found / hdf5 can't open file (OSError), channel not in file (KeyError)
                        continue
                    break
                else:
                    continue
                break
            else:
                # tried all search dirs but didn't find it. Attempt to download.
                raise FileNotFoundError(f'CANNOT FIND {channel}!!')
                # out[channel] = TimeSeries.get(channel, **kwargs) # slow.
        return out

    return channeling_read
