from configparser import ConfigParser
from logging import Formatter, getLogger, FileHandler, StreamHandler
from os import makedirs
from os.path import abspath, join, exists

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


def evalkwargs(**kwargs):
    for key, val in kwargs.items():
        try:
            val = eval(val)  # NOTE, this may be fragile but should allow a lot of flexibility in how we specify options
        except Exception as e:  # NOTE: this could be fragile...
            pass
        kwargs[key] = val
    return kwargs


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
