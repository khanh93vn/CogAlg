'''
    2D version of first-level core algorithm includes frame_blobs, intra_blob (recursive search within blobs), and blob2_P_blob.
    -
    Blob is 2D pattern: connectivity cluster defined by the sign of gradient deviation. Gradient represents 2D variation
    per pixel. It is used as inverse measure of partial match (predictive value) because direct match (min intensity)
    is not meaningful in vision. Intensity of reflected light doesn't correlate with predictive value of observed object
    (predictive value is physical density, hardness, inertia that represent resistance to change in positional parameters)
    -
    Comparison range is fixed for each layer of search, to enable encoding of input pose parameters: coordinates, dimensions,
    orientation. These params are essential because value of prediction = precision of what * precision of where.
    Clustering here is nearest-neighbor only, same as image segmentation, to avoid overlap among blobs.
    -
    Main functions:
    - comp_pixel:
    Comparison between diagonal pixels in 2x2 kernels of image forms derts: tuples of pixel + derivatives per kernel.
    The output is der__t: 2D array of pixel-mapped derts.
    - frame_blobs_root:
    Flood-fill segmentation of image der__t into blobs: contiguous areas of positive | negative deviation of gradient per kernel.
    Each blob is parameterized with summed params of constituent derts, derived by pixel cross-comparison (cross-correlation).
    These params represent predictive value per pixel, so they are also predictive on a blob level,
    thus should be cross-compared between blobs on the next level of search.
    - assign_adjacents:
    Each blob is assigned internal and external sets of opposite-sign blobs it is connected to.
    Frame_blobs is a root function for all deeper processing in 2D alg.
    -
    Please see illustrations:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/blob_params.drawio
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/frame_blobs.png
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/frame_blobs_intra_blob.drawio
'''
from itertools import product
import numpy as np
from utils import kernel_slice_3x3 as ks
# hyper-parameters, set as a guess, latter adjusted by feedback:
ave = 30  # base filter, directly used for comp_r fork
ave_a = 1.5  # coef filter for comp_a fork
aveB = 50
aveBa = 1.5
ave_mP = 100
'''
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
'''

def frame_blobs_root(i__):
    der__t = comp_pixel(i__)  # compare all in parallel -> i__, dy__, dx__, g__, s__
    frame = der__t, I, Dy, Dx, blob_ = [der__t, 0, 0, 0, []]  # init frame as output

    # Flood-fill 1 pixel at a time https://en.wikipedia.org/wiki/Flood_fill:
    yxt_ = [*product(*map(range, der__t[1].shape))]  # set of pixel coordinates to be filled
    root__ = {}  # id map pixel to blob
    perimeter_ = []     # perimeter pixels
    while yxt_:
        if not perimeter_:  # initialize blob
            blob = [frame, None, 0, 0, 0, [], []]  # root (frame), sign, I, Dy, Dx, yxt_, link_ (up-links)
            perimeter_ += [yxt_[0]]

        form_blob(blob, yxt_, perimeter_, root__, der__t)  # https://en.wikipedia.org/wiki/Flood_fill

        if not perimeter_:  # terminate blob
            frame[1] += blob[2]  # I
            frame[2] += blob[3]  # Dy
            frame[3] += blob[4]  # Dx
            blob_ += [blob]

    return frame



def comp_pixel(i__):
    # compute directional derivatives:
    dy__ = (
        (i__[ks.bl] - i__[ks.tr]) * 0.25 +
        (i__[ks.bc] - i__[ks.tc]) * 0.50 +
        (i__[ks.br] - i__[ks.tl]) * 0.25
    )
    dx__ = (
        (i__[ks.tr] - i__[ks.bl]) * 0.25 +
        (i__[ks.mr] - i__[ks.mc]) * 0.50 +
        (i__[ks.br] - i__[ks.tl]) * 0.25
    )
    g__ = np.hypot(dy__, dx__)                          # compute gradient magnitude
    s__ = ave - g__ > 0  # sign is positive for below-average g

    return i__, dy__, dx__, g__, s__


def form_blob(blob, yxt_, perimeter_, root__, der__t):
    # unpack structures
    root, sign, I, Dy, Dx, blob_yxt_, link_ = blob
    i__, dy__, dx__, g__, s__ = der__t
    Y, X = g__.shape

    y, x = perimeter_.pop()  # get coord
    if y < 0 or y >= Y or x < 0 or x >= X: return  # out of bound
    if (y, x) not in yxt_:  # filled
        assert (y, x) in root__ # must have been assigned
        _blob = root__[y, x]
        if _blob not in link_: link_ += [_blob]
        return
    if sign is None: sign = s__[y, x]  # assign sign to new blob
    if sign != s__[y, x]: return   # different sign, stop

    yxt_.remove((y, x))  # remove from yxt_
    root__[y, x] = blob  # assign root, for link forming
    I += i__[ks.mc][y, x]; Dy += dy__[y, x]; Dx += dx__[y, x]; blob_yxt_ += [(y, x)] # update params
    perimeter_ += [(y-1,x), (y,x+1), (y+1,x), (y,x-1)]  # extend perimeter
    if sign: perimeter_ += [(y-1,x-1), (y-1,x+1), (y+1,x+1), (y+1,x-1)]  # ... include diagonals for +blobs

    blob[:] = root, sign, I, Dy, Dx, blob_yxt_, link_ # update blob


if __name__ == "__main__":
    # standalone script, frame_blobs doesn't import from higher modules (like intra_blob).
    # Instead, higher modules will import from frame_blobs and will have their own standalone scripts like below.
    import argparse
    from utils import imread
    # Parse arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpeg')
    args = argument_parser.parse_args()
    image = imread(args.image)

    frame = frame_blobs_root(image)
    # TODO: reusable visualize blobs for higher modules?
