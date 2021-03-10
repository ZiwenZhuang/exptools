
import numpy as np
import multiprocessing as mp
import ctypes
try:
    import torch
except ImportError as e:
    import warnings
    warnings.warn("Cannot import torch, function limited for exptools: " + str(e))

from exptools.collections import namedarraytuple_like


def buffer_from_example(example, leading_dims, share_memory=False):
    ''' Based on given example, build a object in memory that serves as buffer,
    and return the object with the same alignment as example, but with leading dims
    '''
    if example is None:
        return
    try:
        buffer_type = namedarraytuple_like(example)
    except TypeError:  # example was not a namedtuple or namedarraytuple
        return build_array(example, leading_dims, share_memory)
    return buffer_type(*(buffer_from_example(v, leading_dims, share_memory)
        for v in example))


def build_array(example, leading_dims, share_memory=False):
    a = np.asarray(example)
    if a.dtype == "object":
        raise TypeError("Buffer example value cannot cast as np.dtype==object.")
    constructor = np_mp_array if share_memory else np.zeros
    if not isinstance(leading_dims, (list, tuple)):
        leading_dims = (leading_dims,)
    return constructor(shape=leading_dims + a.shape, dtype=a.dtype)


def np_mp_array(shape, dtype):
    size = int(np.prod(shape))
    nbytes = size * np.dtype(dtype).itemsize
    mp_array = mp.RawArray(ctypes.c_char, nbytes)
    return np.frombuffer(mp_array, dtype=dtype, count=size).reshape(shape)


def torchify_buffer(buffer_):
    ''' The array returned will share the same memory as buffer_
    '''
    if buffer_ is None:
        return
    if isinstance(buffer_, np.ndarray):
        return torch.from_numpy(buffer_)
    elif isinstance(buffer_, torch.Tensor):
        return buffer_
    contents = tuple(torchify_buffer(b) for b in buffer_)
    if type(buffer_) is tuple:  # tuple, namedtuple instantiate differently.
        return contents
    return type(buffer_)(*contents)


def numpify_buffer(buffer_):
    ''' The array returned will share the same memory as buffer_
    '''
    if buffer_ is None:
        return
    if isinstance(buffer_, torch.Tensor):
        return buffer_.numpy()
    elif isinstance(buffer_, np.ndarray):
        return buffer_
    contents = tuple(numpify_buffer(b) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def buffer_to(buffer_, device=None):
    ''' Move buffer_ to given device '''
    if buffer_ is None:
        return
    if isinstance(buffer_, torch.Tensor):
        return buffer_.to(device)
    elif isinstance(buffer_, np.ndarray):
        raise TypeError("Cannot move numpy array to device.")
    contents = tuple(buffer_to(b, device=device) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def buffer_method(buffer_, method_name, *args, **kwargs):
    if buffer_ is None:
        return
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):
        return getattr(buffer_, method_name)(*args, **kwargs)
    contents = tuple(buffer_method(b, method_name, *args, **kwargs) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def buffer_func(buffer_, func, *args, **kwargs):
    if buffer_ is None:
        return
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):
        return func(buffer_, *args, **kwargs)
    contents = tuple(buffer_func(b, func, *args, **kwargs) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def get_leading_dims(buffer_, n_dim=1):
    if buffer_ is None:
        return
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):
        return buffer_.shape[:n_dim]
    contents = tuple(get_leading_dims(b, n_dim) for b in buffer_ if b is not None)
    if not len(set(contents)) == 1:
        raise ValueError(f"Found mismatched leading dimensions: {contents}")
    return contents[0]


# def buffer_put(x, loc, y, axis=0, wrap=False):
#     if isinstance(x, (np.ndarray, torch.Tensor)):
#         put(x, loc, y, axis=axis, wrap=wrap)
#     else:
#         for vx, vy in zip(x, y):
#             buffer_put(vx, loc, vy, axis=axis, wrap=wrap)
