"""
's' or 'S' prefix for struct.
'c' or 'C' prefix for class.
'nt' prefix for namedtuple.
"""

from ctypes import *
import numpy as np

from frame_blobs_defs import CBlob, FrameOfBlobs
from utils import imread

class SBlob(Structure):
    _fields_ = [
        ('I', c_double),
        ('G', c_double),
        ('Dy', c_double),
        ('Dx', c_double),
        ('S', c_ulonglong),
        ('sign', c_byte),
        ('box', c_uint * 4),
        ('fopen', c_byte),
    ]

class SFrameOfBlobs(Structure):
    _fields_ = [
        ('I', c_double),
        ('G', c_double),
        ('Dy', c_double),
        ('Dx', c_double),
        ('nblobs', c_ulong),
        ('blobs', POINTER(SBlob)),
    ]

# Load derts2blobs function from C library
derts2blobs = CDLL("frame_blobs.so").derts2blobs
derts2blobs.restype = SFrameOfBlobs

def transfer_data(sframe, dert__, idmap):
    """Transfer data from C structures to custom objects."""
    ntframe = FrameOfBlobs(
        I=sframe.I,
        G=sframe.G,
        Dy=sframe.Dy,
        Dx=sframe.Dx,
        blob_=[],
        dert__=dert__,
    )
    for i in range(sframe.nblobs):
        sblob = sframe.blobs[i]
        y0, yn, x0, xn = sblob.box[:4]
        cblob = CBlob(
            I=sblob.I,
            G=sblob.G,
            Dy=sblob.Dy,
            Dx=sblob.Dx,
            S=sblob.S,
            sign=bool(sblob.sign),
            box=(y0, yn, x0, xn),
            root_dert__=ntframe.dert__,
            adj_blobs=[[], 0, 0],
            fopen=bool(sblob.fopen),
        )
        cblob.mask = (idmap[y0:yn, x0:xn] != i)  # or blob.id


        ntframe.blob_.append(cblob)

    return ntframe

def cwrapped_derts2blobs(dert__):
    dert__ = [*map(lambda a: a.astype('float64'),
                   dert__)]
    height, width = dert__[0].shape
    idmap = np.empty((height, width), 'int64')

    sframe = derts2blobs(*map(lambda d: d.ctypes.data, dert__),
                         height, width, idmap.ctypes.data)

    ntframe = transfer_data(sframe, dert__, idmap)

    return ntframe, idmap, set()
