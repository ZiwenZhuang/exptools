from exptools.logging.tabulate import tabulate
from exptools.logging.console import mkdir_p, colorize
from exptools.logging.autoargs import get_all_parameters
import numpy as np
import os
import os.path as osp
import sys
import datetime
import imageio
# import dateutil.tz
import csv
# import joblib
import json
# import pickle
# import base64
try:
    import torch
except ImportError as e:
    import warnings
    warnings.warn("Cannot import torch, function limited for exptools: " + str(e))
import threading

_tb_available = False
_tb_writer = None
_tb_dump_step = 0 # to synchronize with dump_tabular()
try:
    import tensorboardX
except ImportError as e:
    print("TensorboardX is not available in exptools")
    pass
else:
    _tb_available = True

mp_lock = threading.Lock()

_prefixes = []
_prefix_str = ''

_tabular_prefixes = []
_tabular_prefix_str = ''

_tabular = []

_text_outputs = []
_tabular_outputs = []

_text_fds = {}
_tabular_fds = {}  # key: file_name, value: open file
_tabular_fds_hold = {}
_tabular_header_written = set()

_snapshot_dir = None
_snapshot_mode = 'all'
_snapshot_gap = 1

_log_tabular_only = False
_header_printed = False
_disable_prefix = False

_disabled = False
_tabular_disabled = False


def disable():
    global _disabled
    _disabled = True


def disable_tabular():
    global _tabular_disabled
    _tabular_disabled = True


def enable():
    global _disabled
    _disabled = False


def enable_tabular():
    global _tabular_disabled
    _tabular_disabled = False


def _add_output(file_name, arr, fds, mode='a'):
    if file_name not in arr:
        mkdir_p(os.path.dirname(file_name))
        arr.append(file_name)
        fds[file_name] = open(file_name, mode)


def _remove_output(file_name, arr, fds):
    if file_name in arr:
        fds[file_name].close()
        del fds[file_name]
        arr.remove(file_name)


def push_prefix(prefix):
    _prefixes.append(prefix)
    global _prefix_str
    _prefix_str = ''.join(_prefixes)


def add_text_output(file_name):
    _add_output(file_name, _text_outputs, _text_fds, mode='a')


def remove_text_output(file_name):
    _remove_output(file_name, _text_outputs, _text_fds)


def add_tabular_output(file_name):
    if file_name in _tabular_fds_hold.keys():
        _tabular_outputs.append(file_name)
        _tabular_fds[file_name] = _tabular_fds_hold[file_name]
    else:
        _add_output(file_name, _tabular_outputs, _tabular_fds, mode='a')


def remove_tabular_output(file_name):
    if file_name in _tabular_header_written:
        _tabular_header_written.remove(file_name)
    _remove_output(file_name, _tabular_outputs, _tabular_fds)


def hold_tabular_output(file_name):
    # what about _tabular_header_written?
    if file_name in _tabular_outputs:
        _tabular_outputs.remove(file_name)
        _tabular_fds_hold[file_name] = _tabular_fds.pop(file_name)


def set_snapshot_dir(dir_name):
    os.system("mkdir -p %s" % dir_name)
    global _snapshot_dir
    _snapshot_dir = dir_name


def get_snapshot_dir():
    return _snapshot_dir


def get_snapshot_mode():
    return _snapshot_mode


def set_snapshot_mode(mode):
    if isinstance(mode, int):
        gap = mode
        mode = "gap"
        set_snapshot_gap(gap)
    global _snapshot_mode
    _snapshot_mode = mode


def get_snapshot_gap():
    return _snapshot_gap


def set_snapshot_gap(gap):
    global _snapshot_gap
    _snapshot_gap = gap


def set_log_tabular_only(log_tabular_only):
    global _log_tabular_only
    _log_tabular_only = log_tabular_only


def get_log_tabular_only():
    return _log_tabular_only

def set_disable_prefix(disable_prefix):
    global _disable_prefix
    _disable_prefix = disable_prefix

def get_disable_prefix():
    return _disable_prefix

def tb_scalar(name, data, step):
    if _tb_available:
        """Log a scalar variable."""
        assert _tb_writer is not None
        _tb_writer.add_scalar(tag=name, scalar_value=data, global_step= step)

def tb_text(name, data, step=None):
    ''' data has to be a string
        no need for `step` data
    '''
    if _tb_available:
        assert _tb_writer is not None
        if step is None:
            step = _tb_dump_step
        _tb_writer.add_text(tag=name, text_string=data, global_step= step)

def tb_image(name, data, step=None):
    """ add a image as summary.
    NOTE: data has to be in shape (C, H, W), where C can only be 1, 3 or 4
    """
    if _tb_available:
        assert _tb_writer is not None
        if step is None:
            step = _tb_dump_step
        _tb_writer.add_image(tag=name, img_tensor=data, global_step= step)

def tb_images(name, data, step= None):
    """ add a batch of images as summary
    NOTE: data has to be in shape (N, C, H, W), where C can only be 1, 3 or 4
    """
    if _tb_available:
        assert _tb_writer is not None
        if step is None:
            step = _tb_dump_step
        _tb_writer.add_images(tag= name, img_tensor= data, global_step= step)

def record_image(name, data, itr= None):
    """ NOTE: data must be (H, W) or (3, H, W) or (4, H, W)
    """
    os.system("mkdir -p %s" % os.path.join(_snapshot_dir, "image"))
    filename = os.path.join(_snapshot_dir, "image", "{}-{}.png".format(name, itr))
    if len(data.shape) == 3:
        imageio.imwrite(filename, np.transpose(data, (1,2,0)), format= "PNG")
    else:
        imageio.imwrite(filename, data, format= "PNG")
    tb_image(name, data, itr)

def record_gif(name, data, itr= None, duration= 0.1):
    """ record a series of image as gif into file
    NOTE: data must be a sequence of nparray (H, W) or (3, H, W) or (4, H, W)
    """
    os.system("mkdir -p %s" % os.path.join(_snapshot_dir, "gif"))
    filename = os.path.join(_snapshot_dir, "gif", "{}-{}.gif".format(name, itr))
    if isinstance(data, np.ndarray) or (len(data) > 0 and len(data[0].shape)) == 3:
        imageio.mimwrite(filename, [np.transpose(d, (1,2,0)) for d in data], format= "GIF", duration= duration)
    else:
        imageio.mimwrite(filename, data, format= "GIF", duration= duration)

def log(s, with_prefix=True, with_timestamp=True, color=None):
    if not _disabled:
        out = s
        if with_prefix and not _disable_prefix:
            out = _prefix_str + out
        if with_timestamp:
            now = datetime.datetime.now()  # dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s | %s" % (timestamp, out)
        if color is not None:
            out = colorize(out, color)
        if not _log_tabular_only:
            # Also log to stdout
            print(out)
            for fd in list(_text_fds.values()):
                fd.write(out + '\n')
                fd.flush()
            sys.stdout.flush()
        # add tf text summary if available
        tb_text("log", out)


def record_tabular(key, val, itr= None, *args, **kwargs):
    ''' record scalar 'val' to given 'key', where _tabular_prefix_str will be added
    '''
    # if not _disabled and not _tabular_disabled:
    _tabular.append((_tabular_prefix_str + str(key), str(val)))
    # add tf scalar summary if available
    itr = _tb_dump_step if itr is None else itr
    tb_scalar(key, val, itr)


def push_tabular_prefix(key):
    _tabular_prefixes.append(key)
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)


def pop_tabular_prefix():
    del _tabular_prefixes[-1]
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)


@contextmanager
def prefix(key):
    push_prefix(key)
    try:
        yield
    finally:
        pop_prefix()


@contextmanager
def tabular_prefix(key):
    push_tabular_prefix(key)
    yield
    pop_tabular_prefix()


@contextmanager
def lock():
    """ interface for running logger in multi-threading/processing way """
    mp_lock.acquire()
    yield
    mp_lock.release()

class TerminalTablePrinter:
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


table_printer = TerminalTablePrinter()

_tabular_headers = dict()  # keys are file_names and values are the keys of the header of that tabular file


def dump_tabular(*args, **kwargs):
    ''' write recorded tabular to file.
    
    NOTE: *args, **kwargs options epscifying for 'log' function.
        And 'log' function are used to print tabular.
    '''
    global _tb_dump_step
    _tb_dump_step += 1
    if _tb_available:
        _tb_writer.flush()
    if not _disabled:  # and not _tabular_disabled:
        wh = kwargs.pop("write_header", None)
        if len(_tabular) > 0:
            if _log_tabular_only:
                table_printer.print_tabular(_tabular)
            else:
                for line in tabulate(_tabular).split('\n'):
                    log(line, *args, **kwargs)
            if not _tabular_disabled:
                tabular_dict = dict(_tabular)
                # Also write to the csv files
                # This assumes that the keys in each iteration won't change!
                for tabular_file_name, tabular_fd in list(_tabular_fds.items()):
                    keys = tabular_dict.keys()
                    if tabular_file_name in _tabular_headers:
                        # check against existing keys: if new keys re-write Header and pad with NaNs
                        existing_keys = _tabular_headers[tabular_file_name]
                        if not set(existing_keys).issuperset(set(keys)):
                            joint_keys = set(keys).union(set(existing_keys))
                            tabular_fd.flush()
                            read_fd = open(tabular_file_name, 'r')
                            reader = csv.DictReader(read_fd)
                            rows = list(reader)
                            read_fd.close()
                            tabular_fd.close()
                            tabular_fd = _tabular_fds[tabular_file_name] = open(tabular_file_name, 'w')
                            new_writer = csv.DictWriter(tabular_fd, fieldnames=list(joint_keys))
                            new_writer.writeheader()
                            for row in rows:
                                for key in joint_keys:
                                    if key not in row:
                                        row[key] = np.nan
                            new_writer.writerows(rows)
                            _tabular_headers[tabular_file_name] = list(joint_keys)
                    else:
                        _tabular_headers[tabular_file_name] = keys

                    writer = csv.DictWriter(tabular_fd, fieldnames=_tabular_headers[tabular_file_name])  # list(
                    if wh or (wh is None and tabular_file_name not in _tabular_header_written):
                        writer.writeheader()
                        _tabular_header_written.add(tabular_file_name)
                        _tabular_headers[tabular_file_name] = keys
                    # add NaNs in all empty fields from the header
                    for key in _tabular_headers[tabular_file_name]:
                        if key not in tabular_dict:
                            tabular_dict[key] = np.nan
                    writer.writerow(tabular_dict)
                    tabular_fd.flush()
            del _tabular[:]


def pop_prefix():
    del _prefixes[-1]
    global _prefix_str
    _prefix_str = ''.join(_prefixes)


def save_itr_params(itr, params, name= "", ext= ".pkl"):
    if _snapshot_dir:
        if _snapshot_mode == 'all':
            file_name = osp.join(get_snapshot_dir(), name+'itr_%d'+ext % itr)
        elif _snapshot_mode == 'last':
            # override previous params
            file_name = osp.join(get_snapshot_dir(), name+'params'+ext)
        elif _snapshot_mode == "gap":
            if itr == 0 or (itr + 1) % _snapshot_gap == 0:
                file_name = osp.join(get_snapshot_dir(), name+'itr_%d'+ext % itr)
            else:
                return
        elif _snapshot_mode == 'none':
            return
        else:
            raise NotImplementedError
        torch.save(params, file_name)

def record_tabular_misc_stat(key, values, itr= None, placement='back', pad_nan= False):
    if placement == 'front':
        prefix = ""
        suffix = key
    else:
        prefix = key
        suffix = ""
        if _tb_available:
            prefix += "/"  # Group stats together in Tensorboard.
    if len(values) > 0:
        record_tabular(prefix + "Average" + suffix, np.average(values), itr)
        record_tabular(prefix + "Std" + suffix, np.std(values), itr)
        record_tabular(prefix + "Median" + suffix, np.median(values), itr)
        record_tabular(prefix + "Min" + suffix, np.min(values), itr)
        record_tabular(prefix + "Max" + suffix, np.max(values), itr)
    elif pad_nan:
        record_tabular(prefix + "Average" + suffix, np.nan, itr)
        record_tabular(prefix + "Std" + suffix, np.nan, itr)
        record_tabular(prefix + "Median" + suffix, np.nan, itr)
        record_tabular(prefix + "Min" + suffix, np.nan, itr)
        record_tabular(prefix + "Max" + suffix, np.nan, itr)
