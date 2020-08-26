import ctypes
import numpy as np
import matplotlib.pyplot as plt
from time import time
from frame_blobs_yx import comp_pixel, ave
from utils import imread

frame_blobs_parallel = ctypes.CDLL("frame_blobs.so").frame_blobs_parallel

img = imread('./images/raccoon.jpg')
dert__ = [*map(lambda a: a.astype('float32'),
               comp_pixel(img))]
start_time = time()
height, width = dert__[0].shape
i = dert__[0].ctypes.data
g = dert__[1].ctypes.data
dy = dert__[2].ctypes.data
dx = dert__[3].ctypes.data
bmap = np.empty((height, width), 'uint8')
nblobs = frame_blobs_parallel(i, g, dy, dx, height, width, ave,
                              bmap.ctypes.data)
print(f"{nblobs} blobs formed in {time() - start_time} seconds")
plt.imshow(bmap, 'gray')
plt.show()