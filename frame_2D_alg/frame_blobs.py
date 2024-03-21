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
import weakref
import numpy as np
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
# --------------------------------------------------------------------------------------------------------------
# classes: CBase, CFrame, CG, CBlob, CH

class CBase:
    refs = []
    def __init__(self):
        self._id = len(self.refs)
        self.refs.append(weakref.ref(self))
    def __hash__(self): return self.id
    @property
    def id(self): return self._id
    @classmethod
    def get_instance(cls, _id):
        inst = cls.refs[_id]()
        if inst is not None and inst.id == _id:
            return inst

class CFrame(CBase):
    def __init__(self, i__):
        super().__init__()
        self.i__, self.latuple, self.blob_ = i__, [0, 0, 0, 0], []

    def evaluate(self):
        dert__ = self.comp()
        self.flood_fill(dert__)
        return self

    def comp(self): # compare all in parallel -> i__, dy__, dx__, g__, s__
        # compute directional derivatives:
        dy__ = (
                (self.i__[2:, :-2] - self.i__[:-2, 2:]) * 0.25 +
                (self.i__[2:, 1:-1] - self.i__[:-2, 1:-1]) * 0.50 +
                (self.i__[2:, 2:] - self.i__[:-2, 2:]) * 0.25
        )
        dx__ = (
                (self.i__[:-2, 2:] - self.i__[2:, :-2]) * 0.25 +
                (self.i__[1:-1, 2:] - self.i__[1:-1, :-2]) * 0.50 +
                (self.i__[2:, 2:] - self.i__[:-2, 2:]) * 0.25
        )
        g__ = np.hypot(dy__, dx__)  # compute gradient magnitude, -> separate G because it's not signed, dy,dx cancel out in Dy,Dx
        s__ = ave - g__ > 0  # sign is positive for below-average g

        # convert into dert__:
        y__, x__ = np.indices(self.i__.shape)
        dert__ = dict(zip(
            zip(y__[1:-1, 1:-1].flatten(), x__[1:-1, 1:-1].flatten()),
            zip(self.i__[1:-1, 1:-1].flatten(), dy__.flatten(), dx__.flatten(), g__.flatten(), s__.flatten()),
        ))

        return dert__

    def flood_fill(self, dert__):
        # Flood-fill 1 pixel at a time
        fill_yx_ = list(dert__.keys())  # set of pixel coordinates to be filled (fill_yx_)
        root__ = {}  # map pixel to blob
        perimeter_ = []  # perimeter pixels
        while fill_yx_:  # fill_yx_ is popped per filled pixel, in form_blob
            if not perimeter_:  # init blob
                blob = CBlob(self); perimeter_ += [fill_yx_[0]]

            blob.form(fill_yx_, perimeter_, root__, dert__)  # https://en.wikipedia.org/wiki/Flood_fill

            if not perimeter_:  # term blob
                blob.term()

    def __repr__(self): return f"frame(id={self.id})"

class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster

    def __init__(self, rng=1, fd=0):

        super().__init__()

        self.rng = rng
        self.fd = fd  # fork if flat layers?
        self.n = 0  # external n (last layer n)
        self.area = 0
        self.S = 0  # sparsity: distance between node centers
        self.A = 0, 0  # angle: summed dy,dx in links

        self.Et = []  # external eval tuple, summed from rng++ before forming new graph and appending G.extH
        self.latuple = []  # lateral I,G,M,Ma,L,[Dy,Dx]
        self.iderH = CH()  # summed from PPs
        self.derH = CH()  # nested derH in Gs: [[subH,valt,rdnt,dect]], subH: [[derH,valt,rdnt,dect]]: 2-fork composition layers
        self.node_ = []  # node_t after sub_recursion
        self.link_ = []  # links per comp layer, nest in rng+)der+
        self.roott = []  # Gm,Gd that contain this G, single-layer
        self.box = [np.inf, np.inf, -np.inf, -np.inf]  # y,x,y0,x0,yn,xn
        # graph-external, +level per root sub+:
        self.rim_H = []  # direct links, depth, init rim_t, link_tH in base sub+ | cpr rd+, link_tHH in cpr sub+
        self.extH = CH()  # G-external daggH( dsubH( dderH, summed from rim links
        self.alt_graph_ = []  # adjacent gap+overlap graphs, vs. contour in frame_graphs

    def __bool__(self):  # to test empty
        if self.n: return True
        else: return False
    def __repr__(self): return f"blob(id={self.id})"

class CBlob(CG):
    def __init__(self, root, rng=1):
        super().__init__(rng)
        self.root = root
        self.sign = None
        self.latuple = [0, 0, 0, 0, 0, 0]  # Y, X, I, Dy, Dx, G
        self.dert_ = {}  # keys: (y, x). values: (i, dy, dx, g)
        self.adj_ = []  # adjacent blobs

    def form(self, fill_yx_, perimeter_, root__, dert__):
        y, x = perimeter_.pop()  # pixel coord
        if (y, x) not in dert__: return  # out of bound
        i, dy, dx, g, s = dert__[y, x]
        if (y, x) not in fill_yx_:  # else this is a pixel of adjacent blob
            _blob = root__[y, x]
            if _blob not in self.adj_: self.adj_ += [_blob]
            return
        if self.sign is None: self.sign = s  # assign sign to new blob
        if self.sign != s: return  # different self.sign, stop

        fill_yx_.remove((y, x))
        root__[y, x] = self  # assign root, for link forming
        self.n += 1
        Y, X, I, Dy, Dx, G = self.latuple
        Y += y; X += x; I += i; Dy += dy; Dx += dx; G += g  # update params
        self.latuple = Y, X, I, Dy, Dx, G
        self.dert_[y, x] = i, dy, dx, g  # update elements

        perimeter_ += [(y-1,x), (y,x+1), (y+1,x), (y,x-1)]  # extend perimeter
        if self.sign: perimeter_ += [(y-1,x-1), (y-1,x+1), (y+1,x+1), (y+1,x-1)]  # ... include diagonals for +blobs

    def term(self):
        frame = self.root
        *_, I, Dy, Dx, G = frame.latuple
        *_, i, dy, dx, g = self.latuple
        I += i; Dy += dy; Dx += dx; G += g
        frame.latuple[-4:] = I, Dy, Dx, G
        frame.blob_ += [self]

    @property
    def yx_(self):
        return list(self.dert_.keys())

    @property
    def yx(self):  # as float
        return map(np.mean, zip(*self.yx_))

class CH:  # generic derivation hierarchy of variable nesting

    def __init__(self, nest=0, n=0, Et=None, H=None):

        self.nest = nest  # nesting depth: -1/ ext, 0/ md_, 1/ derH, 2/ subH, 3/ aggH
        self.n = n  # total number of params compared to form derH, summed in comp_G and then from nodes in sum2graph
        self.Et = Et if Et is not None else []  # evaluation tuple: valt, rdnt, normt
        self.H = H if H is not None else []  # hierarchy of der layers or md_

    def __bool__(self):  # to test empty
        if self.n: return True
        else: return False
    '''
    len layer +extt: 2, 3, 6, 12, 24,
    or without extt: 1, 1, 2, 4, 8..: max n of tuples per der layer = summed n of tuples in all lower layers:
    lay1: par     # derH per param in vertuple, layer is derivatives of all lower layers:
    lay2: [m,d]   # implicit nesting, brackets for clarity:
    lay3: [[m,d], [md,dd]]: 2 sLays,
    lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays
    '''

if __name__ == "__main__":

    from utils import imread
    image_file = './images//raccoon_eye.jpeg'
    image = imread(image_file)
    frame = CFrame(image).evaluate()

    # verification/visualization:
    import matplotlib.pyplot as plt
    I, Dy, Dx, G = frame.latuple

    i__ = np.zeros_like(image, dtype=np.float32)
    dy__ = np.zeros_like(image, dtype=np.float32)
    dx__ = np.zeros_like(image, dtype=np.float32)
    g__ = np.zeros_like(image, dtype=np.float32)
    s__ = np.zeros_like(image, dtype=np.float32)
    line_ = []

    for blob in frame.blob_:
        for (y, x), (i, dy, dx, g) in blob.dert_.items():
            i__[y, x] = i; dy__[y, x] = dy; dx__[y, x] = dx; g__[y, x] = g; s__[y, x] = blob.sign
        y, x = blob.yx  # blob center of gravity
        for _blob in blob.adj_:  # show adjacents
            _y, _x = _blob.yx  # _blob center of gravity
            line_ += [((_x, x), (_y, y))]

    plt.imshow(i__, cmap='gray'); plt.show()  # show reconstructed i__
    plt.imshow(dy__,cmap='gray'); plt.show()  # show reconstructed dy__
    plt.imshow(dx__,cmap='gray'); plt.show()  # show reconstructed dx__
    plt.imshow(dx__,cmap='gray'); plt.show()  # show reconstructed dx__
    plt.imshow(g__, cmap='gray'); plt.show()  # show reconstructed g__

    # show blobs and links
    plt.imshow(s__, cmap='gray')
    for line in line_:
        plt.plot(*line, "b-")
    plt.show()