'''
    Intra_blob recursively evaluates each blob for two forks of extended internal cross-comparison and sub-clustering:
    - comp_range: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - vectorize_root: forms roughly edge-orthogonal Ps, evaluated for rotation, comp_slice, etc.
'''
import numpy as np
from frame_blobs import CFrame
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
# filters, All *= rdn:
ave = 50   # cost / dert: of cross_comp + blob formation, same as in frame blobs, use rcoef and acoef if different
aveR = 10  # for range+, fixed overhead per blob
# --------------------------------------------------------------------------------------------------------------
# classes: CIntraBlobFrame (root routine), CIntraBlobLayer (recursive routine)

class CIntraBlobFrame(CFrame):
    def evaluate(self):    # root routine
        super().evaluate()   # run plain frame-blobs once
        for blob in self.blob_:  # init rng+ recursion
            CIntraBlobLayer(blob).evaluate()
        return self

class CIntraBlobLayer(CFrame):
    def __init__(self, blob, rdn=1, rng=1):
        super().__init__(blob.root.i__)  # init params, extra params init below:
        self.root = blob
        self.rdn = rdn
        self.rng = rng

    def evaluate(self):  # recursive evaluation of cross-comp rng+ per blob
        root_blob = self.root
        Y, X, I, Dy, Dx, G = root_blob.vetuple  # unpack vetuple

        if not root_blob.sign: return  # only for below-average G
        if G >= ave*root_blob.n + aveR*self.rdn: return  # eval for comp_r

        root_blob.lay = self
        self.rdn += 1.5; self.rng += 1  # update rdn, rng
        dert__ = self.comp()  # return None if blob is too small
        if not dert__: return   # terminate if blob is too small
        self.flood_fill(dert__)  # else proceed
        del dert__  # to save memory

        for blob in self.blob_:  # recursive eval cross-comp per blob
            CIntraBlobLayer(blob, self.rdn, self.rng).evaluate()
        return self

    def comp(self):   # rng+ comp
        root_blob = self.root
        Y, X = self.i__.shape

        # compute kernel
        ky__, kx__ = compute_kernel(self.rng)

        # loop through root_blob's pixels
        dert__ = {}     # mapping from y, x to dert
        for (y, x), (p, dy, dx, g) in root_blob.dert_.items():
            if y-self.rng < 0 or y+self.rng >= Y or x-self.rng < 0 or x+self.rng >= X: continue # boundary check

            # comparison. i,j: relative coord within kernel 0 -> rng*2+1
            for i, j in zip(*ky__.nonzero()):
                dy += ky__[i, j] * self.i__[y+i-self.rng, x+j-self.rng]    # -rng to get i__ coord
            for i, j in zip(*kx__.nonzero()):
                dx += kx__[i, j] * self.i__[y+i-self.rng, x+j-self.rng]

            g = np.hypot(dy, dx)
            s = ave*(self.rdn + 1) - g > 0

            dert__[y, x] = p, dy, dx, g, s

        return dert__

    def __repr__(self): return f"intra_blob_layer(id={self.id}, root={self.root})"

def compute_kernel(rng):
    # kernel_coefficient = projection_coefficient / distance
    #                    = [sin(angle), cos(angle)] / distance
    # With: distance = sqrt(x*x + y*y)
    #       sin(angle) = y / sqrt(x*x + y*y) = y / distance
    #       cos(angle) = x / sqrt(x*x + y*y) = x / distance
    # Thus:
    # kernel_coefficient = [y / sqrt(x*x + y*y), x / sqrt(x*x + y*y)] / sqrt(x*x + y*y)
    #                    = [y, x] / (x*x + y*y)
    ksize = rng*2+1  # kernel size
    dy, dx = k = np.indices((ksize, ksize)) - rng  # kernel span around (0, 0)
    sqr_dist = dx*dx + dy*dy  # squared distance
    sqr_dist[rng, rng] = 1  # avoid division by 0
    coeff = k / sqr_dist  # kernel coefficient
    coeff[1:-1, 1:-1] = 0  # non-rim = 0

    return coeff

if __name__ == "__main__":
    from utils import imread

    image_file = './images//raccoon_eye.jpeg'
    image = imread(image_file)

    frame = CIntraBlobFrame(image).evaluate()
    # Verification:
    blobQue = frame.blob_
    while blobQue:
        blob = blobQue.pop(0)
        print(f"blob {blob}'s parent is {blob.root}", end="")
        if hasattr(blob, "lay") and blob.lay.blob_:  # if blob is extended with lay
            blob_ = blob.lay.blob_
            print(f", has {len(blob_)} sub-blob{'' if len(blob_) == 1 else 's'}")
            if blob_: blobQue += blob_
        else: print()
        # else un-extended blob, skip