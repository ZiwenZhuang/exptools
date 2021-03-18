from contextlib import contextmanager
from exptools.logging.tabulate import tabulate
from exptools.logging.console import mkdir_p, colorize
from exptools.logging.autoargs import get_all_parameters
import numpy as np
from collections import OrderedDict, defaultdict
import os, shutil
import os.path as osp
import sys
import datetime
import pandas as pd
import imageio
import csv
import threading
import json

_tb_avaliable = False
tb_writer = None
try:
    import tensorboardX
except ImportError as e:
    print("TensorboardX is not available in exptools, logging might be limited")
else:
    _tb_avaliable = True

class Logger():
    """ The interface to handle all logging operations (if you are using this library).
    Current logging modalities: text, scalar, image, gif, pointcloud/mesh, 
    All modalities can be logged in batch, which means the datas should be able to be indexed as data[i]
    NOTE: all filename and paths (except self.log_dir) are relative paths related to self.log_dir
    """
    def __init__(self,
            log_dir, # The abspath of where all log files are put
            refresh= True, # if you don't want to resume your experiment, this will remove everything in log_dir
        ):
        self.log_dir = osp.abspath(log_dir)
        mkdir_p(self.log_dir)
        self.mp_lock = threading.Lock()

        # cleaning the log_dir if necessary
        if refresh:
            for filename in os.listdir(self.log_dir):
                _fp = os.path.join(self.log_dir, filename)
                try:
                    if os.path.isfile(_fp) or os.path.islink(_fp):
                        os.unlink(_fp)
                    elif os.path.isdir(_fp):
                        shutil.rmtree(_fp)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (_fp, e))


        # start building all logging stuff
        self.tb_writer = None if not _tb_avaliable else tensorboardX.SummaryWriter(logdir= self.log_dir)
        
        self._text_prefix = [] # a stack to set prefix
        self._text_files = {} # dict of {filename:file_descriptor}
        self._text_default_file = None

        # assuming current scalar data can be handled by cluster memory (otherwise, solve later)
        self._scalar_prefix = [] # a stack to set prefix
        self._scalar_data = {} # a dict of {filename:pandas_dataframe}
        self._scalar_default_file = None

    def push_text_prefix(self, prefix: str):
        self._text_prefix.append(prefix)
    def pop_text_prefix(self):
        self._text_prefix.pop(-1)
        
    def push_scalar_prefix(self, prefix: str):
        self._scalar_prefix.append(prefix)
    def pop_scalar_prefix(self):
        self._scalar_prefix.pop(-1)

    def push_prefix(self, prefix: str):
        self.push_text_prefix(prefix)
        self.push_scalar_prefix(prefix)
    def pop_prefix(self):
        self.pop_text_prefix()
        self.pop_scalar_prefix()

    def add_text_output(self, filename: str):
        if not self._text_default_file:
            self._text_default_file = filename
        self._text_files[filename] = open(osp.join(self.log_dir, filename), mode= "a")
    def remove_text_output(self, filename):
        if filename == self._text_default_file:
            print(colorize(
                "Warning: You are removing default text output",
                color= "yellow",
            ))
            self._text_default_file = None
        self._text_files[filename].close()
        self._text_files.pop(filename)
    @contextmanager
    def additional_text_output(self, filename):
        self.add_text_output(filename)
        yield
        self.remove_text_output(filename)

    def redirect_stdout_to_text_output(self):
        """ NOTE: You have to add_text_output before calling this method
        """
        sys.stdout = self._text_files[self._text_default_file]
    def redirect_stdout_to_console(self):
        sys.stdout = sys.__stdout__

    def save_param_dict(self, param, filename):
        assert isinstance(param, dict)
        with open(osp.join(self.log_dir, filename), "w") as fd:
            json.dump(param, fd, indent= 4)
    
    def add_scalar_output(self, filename: str):
        if not self._scalar_default_file:
            self._scalar_default_file = filename
        self._scalar_data[filename] = pd.DataFrame().append({}, ignore_index= True)
    def remove_scalar_output(self, filename= None):
        if filename is None: filename = self._scalar_default_file
        if filename == self._scalar_default_file:
            print(colorize(
                "Warning: You are removing default scalar output",
                color= "yellow",
            ))
            self._scalar_default_file = None
        self._scalar_data[filename].to_csv(osp.join(self.log_dir, filename), index= False)
        self._scalar_data.pop(filename)
    @contextmanager
    def additional_scalar_output(self, filename):
        self.add_scalar_output(filename)
        yield
        self.remove_scalar_output(filename)

    def log_text(self, data, step,
            filename= None,
            with_prefix= True,
            with_timestamp=True,
            color=None
        ):
        if filename is None: filename = self._text_default_file

        out = data
        if with_prefix: 
            for p in self._text_prefix:
                out = p + out
        if with_timestamp:
            now = datetime.datetime.now()  # dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s | %s" % (timestamp, out)
        if color is not None:
            out = colorize(out, color)

        print(out)
        self._text_files[filename].write(out + "\n")
        self._text_files[filename].flush()
        if not self.tb_writer is None:
            self.tb_writer.add_text("text", out, step)

    def log_scalar(self, tag, data, step= None, filename= None, with_prefix= True, **kwargs):
        """
        @Args:
            tag: string;
            data: a number (not array)
            step: a int of the iteration number. If `filename` provided, 
                you need to give proper `step` of current `filename`
        """
        if filename is None: filename = self._scalar_default_file
        if with_prefix:
            for p in self._scalar_prefix:
                tag = p + tag
        # maintain pandas DataFrame
        df_len = len(self._scalar_data[filename])
        if step is None: step = df_len - 1
        if not tag in self._scalar_data[filename]:
            self._scalar_data[filename][tag] = np.nan
        try:
            self._scalar_data[filename].loc[step][tag] = data
        except KeyError as e:
            print(colorize("KeyError: {}".format(e), color= "red"))
            print(colorize("You might forget to dump_scalar for your scalar file, check demo script please", color= "yellow"))
            exit(-1)
        # tensorboardX API
        if not self.tb_writer is None:
            self.tb_writer.add_scalar(tag, data, step)

    def log_scalar_batch(self, tag, data, step, filename= None, **kwargs):
        """ Record a batch of data with several statictis
        data: a array of numbers np.array is better
        """
        if not isinstance(data, np.ndarray): data = np.array(data)
        if len(data) > 0:
            self.log_scalar(tag + "/Average", np.nanmean(data), step, filename, **kwargs)
            self.log_scalar(tag + "/Std", np.nanstd(data), step, filename, **kwargs)
            self.log_scalar(tag + "/Max", np.nanmax(data), step, filename, **kwargs)
            self.log_scalar(tag + "/Min", np.nanmin(data), step, filename, **kwargs)
            self.log_scalar(tag + "/Len", np.count_nonzero(~np.isnan(data)), step, filename, **kwargs)

    def dump_scalar(self, filename= None):
        """ In order to reflect the scalar data to the file and data loss due to program crash
        we write current scalar dataframe to csv file
        """
        if filename is None: filename = self._scalar_default_file
        
        self._scalar_data[filename].to_csv(osp.join(self.log_dir, filename), index= False)
        self.log_text("Dumping scalar data for {}".format(filename), len(self._scalar_data[filename]))
        print(tabulate( self._scalar_data[filename].iloc[-1].items() ))
        self._scalar_data[filename] = self._scalar_data[filename].append({}, ignore_index= True)

    def __old_dump_scalar(self, filename= None):
        """ Due to csv feature, you need to dump scalar to csv file. You can 
        also specify the filename for which file you are dumping to
        """
        if filename is None: filename = self._scalar_default_file

        current_reader = csv.reader(self._scalar_files[filename])
        # print current data
        if len(current_reader) == 0:
            text_step = 0
        else:
            text_step = len(current_reader) - 1
        self.log_text("Dumping scalar data for {}".format(filename), text_step)
        print(tabulate( self._scalar_current_data[filename].items() ))
        # check current file keys, and determine whether to rewrite the entire file
        if len(current_reader) > 0:
            old_keys = next(current_reader)
            current_keys = list(self._scalar_current_data[filename].keys()) # a copy of keys
            del current_reader
            # checking keys
            key_unchanged = len(old_keys) == len(current_keys)
            for csv_k, data_k in zip(old_keys, current_keys):
                if csv_k != data_k:
                    key_unchanged = False; break
            if key_unchanged:
                # keep writing
                current_writer = csv.DictWriter(self._scalar_files[filename], current_keys)
                current_writer.writerow(self._scalar_current_data[filename])
                self._scalar_files[filename].flush()
            else:
                # rewrite the entire csv file (hope this never comes)
                keys_to_add = []
                for key in old_keys: # if current_keys < old_keys
                    if not key in current_keys:
                        self._scalar_current_data[filename][key] = np.nan
                with open(osp.join(self.log_dir, self._TEMP_CSV_FILENAME), "w") as new_fd:
                    old_reader = csv.DictReader(self._scalar_files[filename])
                    new_writer = csv.DictWriter(new_fd, fieldnames= list(self._scalar_current_data[filename].keys()))
                    # rewrite old data
                    for row in old_reader:
                        row = defaultdict(lambda:np.nan, **row) # if current_keys > old_keys
                        new_writer.writerow(row)
                    # write new data
                    new_writer.writerow(self._scalar_current_data[filename])
                    new_fd.flush()
                # replace file descriptor
                self._scalar_files[filename].close()
                os.remove(osp.join(self.log_dir, filename)) # NOTE: currently, `filename` is invalid filename
                os.rename(
                    osp.join(self.log_dir, self._TEMP_CSV_FILENAME),
                    osp.join(self.log_dir, filename),
                )
                self._scalar_files[filename] = open(osp.join(self.log_dir, filename))
        else:
            # new file, write directly
            del current_reader
            file_writer = csv.DictWriter(self._scalar_files[filename], fieldnames= list(self._scalar_files[filename].keys()))
            file_writer.writeheader()
            file_writer.writerow(self._scalar_current_data[filename])
        # clear out current data (buffer)
        for k in self._scalar_current_data[filename].keys():
            self._scalar_current_data[filename][k] = np.nan
    
    def log_image(self, tag, data, step, **kwargs):
        """ NOTE: data must be (H, W) or (3, H, W) or (4, H, W) from 0-255 uint8
        """
        mkdir_p(osp.join(self.log_dir, "image"))
        filename = osp.join(self.log_dir, "image", "{}-{}.png".format(tag, step))
        if len(data.shape) == 3:
            imageio.imwrite(filename, np.transpose(data, (1,2,0)), format= "PNG")
        else:
            imageio.imwrite(filename, data, format= "PNG")
        if not self.tb_writer is None:
            self.tb_writer.add_image(tag, data, step)

    def log_gif(self, tag, data, step, duration= 0.1, **kwargs):
        """ record a series of image as gif into file
        NOTE: data must be a sequence of nparray (H, W) or (3, H, W) or (4, H, W) from 0-255 uint8
        """
        mkdir_p(osp.join(self.log_dir, "gif"))
        filename = osp.join(self.log_dir, "gif", "{}-{}.gif".format(tag, step))
        if isinstance(data, np.ndarray) or (len(data) > 0 and len(data[0].shape)) == 3:
            imageio.mimwrite(filename, [np.transpose(d, (1,2,0)) for d in data], format= "GIF", duration= duration)
        else:
            imageio.mimwrite(filename, data, format= "GIF", duration= duration)
        # TensorboardX does not support this yet

    def __del__(self):
        try:
            for _, v in self._text_files.items():
                v.close()
        except:
            print(colorize("Exceptions when closing text logger", color= "yellow"))
        try:
            for f, d in self._scalar_data.items():
                d.to_csv(osp.join(self.log_dir, f), index= False)
        except:
            print(colorize("Exceptions when closing scalar logger", color= "yellow"))
        try:
            if not self.tb_writer is None:
                self.tb_writer.close()
        except:
            print(colorize("Exceptions when closing tensorboardX writer", color= "yellow"))

    # >>>>>>>>> The followings are APIs for other experiment platforms <<<<<<<<    

        
    



