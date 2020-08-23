'''
Another draft of blob-parallel version of frame_blobs
'''

from collections import deque
from class_cluster import ClusterStructure, NoneType
from itertools import minmax

class CBlob(ClusterStructure):
    # Derts
    I = int
    G = int
    Dy = int
    Dx = int
    S = int
    # other data
    box = list
    sign = NoneType
    dert_coord_ = set  # let derts' id be their coords
    root_dert__ = object
    adj_blobs = list
    fopen = bool

def frame_blobs_parallel(dert__):
    height, width = dert__.shape[-2:]
    id_map = np.full(shape[-2:], -1, 'uint64')  # blob's id per dert, initialized with -1
    blob_ = []
    for y in range(height):
        for x in range(width):
            if id_map[y, x] == -1:  # ignore filled/clustered derts (blob id != -1)
                # initialize new blob
                blob = CBlob(I=dert[0], G=dert[1], Dy=dert[2], Dx=dert[3],
                             sign=dert[1] > 0, root_dert__=dert__)
                blob_.append(blob)

                # flood fill the blob, start from current position
                unfilled_derts = deque((y, x))
                while unfilled_derts:
                    y1, x1 = unfilled_derts.popleft()

                    # add dert to blob
                    blob.dert_coord_.add((y1, x1))  # add dert coordinate to blob
                    id_map[y1, x1] = blob.id  # add blob ID to each dert
                    blob.I += dert[0, y1, x1]
                    blob.G += dert[1, y1, x1]
                    blob.Dy += dert[2, y1, x1]
                    blob.Dx += dert[3, y1, x1]
                    blob.S += 1

                    # determine neighbors' coordinates, 4 for -, 8 for +
                    if blob.sign:   # include diagonals
                        adj_dert_coords = [(y1 - 1, x1 - 1), (y1 - 1, x1),
                                           (y1 - 1, x1 + 1), (y1, x1 + 1),
                                           (y1 + 1, x1 + 1), (y1 + 1, x1),
                                           (y1 + 1, x1 - 1), (y1, x1 - 1)]
                    else:
                        adj_dert_coords = [(y1 - 1, x1), (y1, x1 + 1),
                                           (y1 + 1, x1), (y1, x1 - 1)]

                    # search through neighboring derts
                    for y2, x2 in adj_dert_coords:
                        # check if image boundary is reached
                        if (y2 < 0 or y2 >= height or
                            x2 < 0 or x2 >= width):
                            blob.fopen = True
                        # check if same-signed
                        elif blob.sign == dert__[1, y2, x2] > 0:
                            assert id_map[y2, x2] == -1  # should be unfilled
                            unfilled_derts.append((y2, x2))
                        # else assign adjacents
                        else:
                            # TODO: assign adjacents
                            pass
                # terminate blob
                y_coords, x_coords = zip(*blob.dert_coord_)
                blob.box = (
                    *minmax(y_coords),  # y0, yn
                    *minmax(x_coords),  # x0, xn
                )
                # got a set of coordinates, no need for mask?
