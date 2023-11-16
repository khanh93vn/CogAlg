import numpy as np
from frame_blobs import ave
from utils import kernel_slice_3x3 as ks

class GenericCycle:
    def __init__(self, node_, link_,            # nodes and links
                 sub=False, agg=False,          # if True perform corresponding recursion
                 verbose=False):
        self.node_ = node_
        self.link_ = link_
        self.do_sub = sub
        self.do_agg = agg
        self.verbose = verbose

    def evaluate(self):
        # generic cycle: xcomp -> cluster -> sub+ eval ) agg+ eval
        self.xcmp()
        self.cluster()
        if self.do_sub: self.sub_rec()
        if self.do_agg: self.agg_rec()

    def xcmp(self):         # to be overwritten by specific sequence
        raise NotImplementedError

    def cluster(self):      # to be overwritten by specific sequence
        raise NotImplementedError

    def sub_rec(self):      # to be overwritten by specific sequence
        raise NotImplementedError

    def agg_rec(self):      # to be overwritten by specific sequence
        raise NotImplementedError


# Demonstration, example of FrameBlobs:
class FrameBlobs(GenericCycle):

    def xcmp(self):
        i__ = self.node_   # node_ is 2d in this special case, and no link_
        # compute directional derivatives:
        self.dy__ = (
                (i__[ks.bl] - i__[ks.tr]) * 0.25 +
                (i__[ks.bc] - i__[ks.tc]) * 0.50 +
                (i__[ks.br] - i__[ks.tl]) * 0.25
        )
        self.dx__ = (
                (i__[ks.tr] - i__[ks.bl]) * 0.25 +
                (i__[ks.mr] - i__[ks.mc]) * 0.50 +
                (i__[ks.br] - i__[ks.tl]) * 0.25
        )
        self.g__ = np.hypot(self.dy__, self.dx__)  # compute gradient magnitude

    def cluster(self):
        self.s__ = ave - self.g__ > 0  # sign is positive for below-average g
        # https://en.wikipedia.org/wiki/Flood_fill:
        self.flood_fill()
        self.assign_adjacents()  # forms adj_blobs per blob in adj_pairs, or keep list of adj_pairs?

    def flood_fill(self):
        # (flood fill goes here, along with adj_pairs assignments
        self.adj_pairs = []

    def assign_adjacents(self):
        if getattr(self, 'adj_pairs', None) is None: return     # return if no self.adj_pairs
        # (adjacent pairs assignments (links) goes here)
        del self.adj_pairs  # no longer needed

    def sub_rec(self):  # sub recursion
        # (do intra_blob here)
        pass

    def agg_rec(self):
        # (initialize and run another GenericCycle, with blobs as nodes and adj_pairs as links)
        pass