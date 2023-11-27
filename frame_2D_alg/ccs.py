# generic cycle: xcomp -> prune (links) -> form (clusters) -> sub+ eval ) agg+ eval
import numpy as np
from frame_blobs import ave
from utils import kernel_slice_3x3 as ks
from collections import deque, namedtuple
from itertools import product

from utils import imread

class CCBase:
    graphT = namedtuple("graphT", "node_ link_")
    cnodeT = None
    def __init__(self, igraph, sub=False):
        # inputs
        self.igraph = igraph      # input graph
        self.do_sub = sub

        # outputs
        self.cgraph = None

    def evaluate(self):
        # generic cycle: xcomp -> prune (links) -> form (clusters) -> sub+ eval ) agg+ eval
        self.cgraph = self.graphT([], [])  # clustered object

        self.xcmp()
        self.prune()
        self.form()
        if self.do_sub: self.sub()

    def xcmp(self):
        raise NotImplementedError

    def prune(self):
        raise NotImplementedError

    def form(self):
        raise NotImplementedError

    def sub(self):
        raise NotImplementedError


# Demonstration example of FrameBlobs:
class FrameBlobs(CCBase):
    UNFILLED = -1
    EXCLUDED = -2
    cnodeT = blobT = namedtuple("blobT", "id sign dert root fopen")
    cnodeT.G = property(lambda self: np.hypot(*self.dert[1:3]))   # G from Dy, Dx

    def xcmp(self):
        self.i__ = self.igraph.node_
        self.dy__ = np.zeros_like(self.i__, float)
        self.dx__ = np.zeros_like(self.i__, float)
        self.g__ = np.zeros_like(self.i__, float)

        for _idx, idx, (disty, distx) in self.igraph.link_:
            d__ = self.i__[idx] - self.i__[_idx]
            # compute directional derivatives:
            dist2 = disty**2 + distx**2
            dy__ = d__ * disty / dist2
            dx__ = d__ * distx / dist2
            self.dy__[_idx] += dy__; self.dy__[idx] += dy__
            self.dx__[_idx] += dx__; self.dx__[idx] += dx__
        self.g__ = np.hypot(self.dy__, self.dx__)

    def prune(self):
        self.s__ = ave - self.g__ > 0   # sever (implicit) links of different-signed neighbors

    def form(self):
        blob_, adjt_ = self.cgraph
        Y, X = self.s__.shape
        idx = 0
        self.idx__ = np.full((Y, X), -1, 'int32')
        for __y, __x in product(range(Y), range(X)):
            if self.idx__[__y, __x] != self.UNFILLED: continue    # ignore filled/clustered derts
            sign = self.s__[__y, __x]
            fopen = False

            # flood fill the blob, start from current position
            fillQ = deque([(__y, __x)])
            while fillQ:
                _y, _x = fillQ.popleft()
                self.idx__[_y, _x] = idx
                # neighbors coordinates, 4 for -, 8 for +
                adj_yx_ = [ (_y-1,_x), (_y,_x+1), (_y+1,_x), (_y,_x-1) ]
                if sign: adj_yx_ += [(_y-1,_x-1), (_y-1,_x+1), (_y+1,_x+1), (_y+1,_x-1)] # include diagonals
                # search neighboring derts:
                for y, x in adj_yx_:
                    if (y, x) in fillQ: continue
                    if not (0<=y<Y and 0<=x<X) or self.idx__[y, x] == self.EXCLUDED: fopen = True    # image boundary is reached
                    elif self.idx__[y, x] == self.UNFILLED:    # pixel is filled
                        if self.s__[y, x] == sign: fillQ += [(y, x)]     # add to queue if same-sign dert
                    elif self.s__[y, x] != sign:            # else check if same-signed
                        adjt = (self.idx__[y, x], idx)
                        if adjt not in adjt_: adjt_ += [adjt]
            # terminate blob
            msk = (self.idx__ == idx)
            blob = self.blobT(
                id=idx,
                sign=sign,
                root=-1,
                fopen=fopen,
                dert=np.array([
                    self.i__[msk].sum(),            # I
                    self.dy__[msk].sum(),           # Dy
                    self.dx__[msk].sum()]))         # Dx
            blob_ += [blob]
            idx += 1

    def sub(self):
        pass

    def intra(self, root_blob):
        pass

if __name__ == "__main__":
    image = imread("images/raccoon_eye.jpeg")
    # form link
    alc, fst, lst = slice(None), slice(None, -1), slice(1, None)
    igraph = CCBase.graphT(
        image,
        [
            ((fst, alc), (lst, alc), (1, 0)),
            ((alc, fst), (alc, lst), (0, 1)),
            ((fst, fst), (lst, lst), (1, 1)),
            ((fst, lst), (lst, fst), (1, -1)),
        ],
    )

    frame_blobs = FrameBlobs(igraph)
    frame_blobs.evaluate()
    import matplotlib.pyplot as plt
    img = np.full((frame_blobs.s__.shape), 128, 'uint8')
    for blob in frame_blobs.cgraph.node_:
        msk = frame_blobs.idx__ == blob.id
        img[msk] = 255 * blob.sign
    plt.imshow(img, cmap='gray')
    plt.show()
