import numpy as np
import zarr

def line_count(filename):
    def _make_gen(reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024*1024)

    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum( buf.count(b'\n') for buf in f_gen )

def pad_1d(lst, length, pad_idx=0):
    """ Pad over axis 0. """
    return np.pad(lst, (0, length-len(lst)), 'constant', constant_values=pad_idx)

def pad_2d(lst_of_lst, length, pad_idx=0):
    """ Pad over axis 0, 1. """
    # length: tuple
    res = np.full(length, pad_idx)
    for i, vec in enumerate(lst_of_lst):
        res[i, :len(vec)] = vec
    return res

def append_storage(storage, append_len):
    storage.append(zarr.zeros((append_len, *storage.shape[1:])))

def resize_storage(storage, len):
    storage.resize(len, *storage.shape[1:])