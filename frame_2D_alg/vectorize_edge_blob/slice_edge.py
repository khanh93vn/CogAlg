import numpy as np
from collections import defaultdict
from math import atan2, cos, floor, pi, sin
import sys
sys.path.append("..")
from frame_blobs import CBase, CH, imread   # for CP
from intra_blob import CsubFrame

'''
In natural images, objects look very fuzzy and frequently interrupted, only vaguely suggested by initial blobs and contours.
Potential object is proximate low-gradient (flat) blobs, with rough / thick boundary of adjacent high-gradient (edge) blobs.
These edge blobs can be dimensionality-reduced to their long axis / median line: an effective outline of adjacent flat blob.
-
Median line can be connected points that are most equidistant from other blob points, but we don't need to define it separately.
An edge is meaningful if blob slices orthogonal to median line form some sort of a pattern: match between slices along the line.
These patterns effectively vectorize representation: they represent match and change between slice parameters along the blob.
-
This process is very complex, so it must be selective. Selection should be by combined value of gradient deviation of edge blobs
and inverse gradient deviation of flat blobs. But the latter is implicit here: high-gradient areas are usually quite sparse.
A stable combination of a core flat blob with adjacent edge blobs is a potential object.
'''
octant = 0.3826834323650898  # radians per octant
aveG = 10  # for vectorize
ave_g = 30  # change to Ave from the root intra_blob?
ave_dangle = .8  # vertical difference between angles: -1->1, abs dangle: 0->1, ave_dangle = (min abs(dangle) + max abs(dangle))/2,

class CsliceEdge(CsubFrame):

    class CEdge(CsubFrame.CBlob):     # replaces CBlob after definition

        def term(blob):     # an extension to CsubFrame.CBlob.term(), evaluate for vectorization right after rng+ in intra_blob
            super().term()
            if not blob.sign and blob.G > aveG * blob.root.rdn:
                blob.vectorize()

        def vectorize(blob):        # to be overridden in higher modules (comp_slice, agg_recursion)
            blob.slice_edge_hough()

        def slice_edge_hough(edge):
            # DEBUG:
            ratan = 18
            athr = 6
            rata = ratan/pi
            ratr = 8
            rthr = 4

            from copy import copy
            import matplotlib.pyplot as plt
            yx_ = np.array(edge.yx_)
            y0, x0 = yx0 = yx_.min(axis=0) - 1

            # show edge-blob
            shape = yx_.max(axis=0) - yx0 + 2
            mask_nonzero = tuple(zip(*(yx_ - yx0)))
            mask = np.zeros(shape, bool)
            mask[mask_nonzero] = True
            plt.cla()
            plt.imshow(mask, cmap='gray', alpha=0.5)
            plt.title(f"area = {edge.area}")

            # find carrying lines
            lined = defaultdict(list)
            vald = {}
            for (y, x), (i, gy, gx, g) in edge.dert_.items():
                c, s = gx/g, gy/g
                a = atan2(-c, s) if c < 0 else atan2(c, -s)
                assert 0 <= a <= pi

                ab = round(a*rata)
                for _ab in range(ab - athr, ab + athr + 1):
                    if _ab < 0: _ab += ratan
                    if _ab >= ratan: _ab -= ratan
                    a = _ab / rata
                    _s, _c = sin(a), cos(a)
                    r = y*_s + x*_c
                    rb = round(r*ratr)
                    for _rb in range(rb - rthr, rb + rthr + 1):
                        lined[_ab, _rb] += [(y, x)]
                        vald[(_ab, _rb), (y, x)] = 5*abs(c*_s - s*_c) - abs(r - _rb/ratr)
                plt.quiver(x-x0, y-y0, c, -s, scale=100, headwidth=1, headlength=2)

            # segment lines into segments
            segplotd = {}
            segd = defaultdict(list)
            rootd = defaultdict(list)
            endd = {}
            for (ab, rb), yx_ in lined.items():
                a, r = ab/rata, rb/ratr
                s, c = sin(a), cos(a)

                i = 0
                while yx_:
                    fill_ = [yx_[0]]
                    while fill_:
                        y, x = fill_.pop()
                        if (y, x) not in yx_: continue
                        segd[ab, rb, i] += [(y, x)]
                        yx_.remove((y, x))
                        fill_ += [(y-1,x-1),(y-1,x),(y-1,x+1),(y,x+1),(y+1,x+1),(y+1,x),(y+1,x-1),(y,x-1)]

                    # draw segment
                    pmin, pmax = np.inf, -np.inf
                    for y, x in segd[ab, rb, i]:
                        p = y*c - x*s
                        if p < pmin:
                            pmin = p
                            yxmin = y, x
                        if p > pmax:
                            pmax = p
                            yxmax = y, x

                    y1, x1 = project(*yxmin, s, c, r) - yx0
                    y2, x2 = project(*yxmax, s, c, r) - yx0
                    endd[ab, rb, i] = ((y1, x1), (y2, x2))
                    u, v = -s*0.5, c*0.5
                    y1, x1, y2, x2 = y1-v, x1-u, y2+v, x2+u
                    segplotd[ab, rb, i], = plt.plot([x1, x2], [y1, y2], 'b-')

                    for y, x in segd[ab, rb, i]:
                        vald[(ab, rb), (y, x)] += 1/(1 + abs(pmax - pmin))
                        rootd[y, x] += [(ab, rb, i)]

                    i += 1

            plt.ion()
            plt.show()
            plt.pause(0.001)

            del lined
            pruned_rootd = {yx:copy(root_) for yx, root_ in rootd.items()}

            thres_dec = 0.95
            min_thres = 0.0
            thres = 1.0
            # prune segments
            while True:
                rdn = 0
                segvoted = defaultdict(int)
                for y, x in pruned_rootd:  # compete for yx?
                #     if not pruned_rootd[y, x]:
                #         pruned_rootd[y, x] = copy(rootd[y, x])
                #         for ab, rb, i in pruned_rootd[y, x]:
                #             segplotd[ab, rb, i].set_alpha(1.0)
                    if len(pruned_rootd[y, x]) <= 1: continue

                    if len(pruned_rootd[y, x]) < 10:
                        root_ = list(pruned_rootd[y, x])
                        olp_ = set()
                        while root_:
                            ab, rb, i = root_.pop()
                            (y1, x1), (y2, x2) = endd[ab, rb, i]
                            a, r = ab/rata, rb/ratr
                            s, c = sin(a), cos(a)
                            for _ab, _rb, _i in root_:
                                (_y1, _x1), (_y2, _x2) = endd[_ab, _rb, _i]
                                _a, _r = _ab/rata, _rb/ratr
                                _s, _c = sin(_a), cos(_a)
                                d1 = y1*_s + x1*_c - _r
                                d2 = y2*_s + x2*_c - _r
                                _d1 = _y1*s + _x1*c - r
                                _d2 = _y2*s + _x2*c - r
                                if d1*d2 >= 0 and abs(d1)+abs(d2) > 2.0: continue
                                if _d1*_d2 >= 0 and abs(_d1)+abs(_d2) > 2.0: continue
                                if vald[(ab, rb), (y, x)] <= vald[(_ab, _rb), (y, x)]:
                                    olp_.add((ab, rb, i))
                                if vald[(ab, rb), (y, x)] >= vald[(_ab, _rb), (y, x)]:
                                    olp_.add((_ab, _rb, _i))
                    else: olp_ = pruned_rootd[y, x]

                    sorted_olp_ = sorted(olp_, key=lambda root: vald[root[:2], (y, x)], reverse=True)
                    nkeep = int(len(sorted_olp_)/2)
                    for ab, rb, i in sorted_olp_[nkeep:]:
                        segvoted[ab, rb, i] -= 1/len(segd[ab, rb, i])
                    rdn += max(0, len(olp_) - 1)

                if rdn == 0:
                    print("done")
                    break
                else: print(rdn)

                # prune
                prune_ = sorted(segvoted, key=lambda k: segvoted[k])
                for ab, rb, i in prune_:
                    if segvoted[ab, rb, i] >= -thres: break
                    for y, x in segd[ab, rb, i]:
                        if (ab, rb, i) in pruned_rootd[y, x]:
                            pruned_rootd[y, x].remove((ab, rb, i))
                    segplotd[ab, rb, i].set_alpha(0.0)

                plt.draw()
                plt.pause(0.001)

                if thres > min_thres: thres *= thres_dec

            pruned_segd = defaultdict(list)
            for y, x in pruned_rootd:
                # if not pruned_rootd[y, x]:  # if all roots is lost: re-init
                #     candidates = sorted(
                #         [(ab, rb, i) for ab, rb, i in rootd[y, x]
                #          if not sum([len(pruned_rootd[_y, _x]) for _y, _x in segd[ab, rb, i]])],
                #         key=lambda root: vald[root[:2], (y, x)],
                #         reverse=True,
                #     )
                #     if candidates:
                #         ab, rb, i = candidates[0]
                #         segplotd[ab, rb, i].set_alpha(1.0)
                #         for _y, _x in segd[ab, rb, i]:
                #             pruned_rootd[_y, _x] = [(ab, rb, i)]
                for ab, rb, i in pruned_rootd[y, x]:
                    pruned_segd[ab, rb, i] += [(y, x)]

            plt.draw()
            plt.pause(0.001)

            while input() != 'q':
                plt.draw()
                plt.pause(0.001)

            # form Ps
            edge.P_ = []
            for (ab, rb, i), yx_ in pruned_segd.items():
                a, r = ab/rata, rb/ratr
                s, c = sin(a), cos(a)
                _y, _x = np.array(yx_).mean(axis=0)
                y, x = project(_y, _x, s, c, r)     # project _y, _x onto the line cx + sy = r

                axis = c, -s  # axis of the line
                dert = interpolate2dert(edge, y, x)  # center
                if dert is not None:
                    edge.P_ += [CP(edge, (y, x), axis, dert)]

        def slice_edge(edge):
            edge.P_ = [CP(edge, yx, axis) for yx, axis in edge.select_max()]  # P_ is added dynamically, only edge-blobs have P_
            edge.P_ = sorted(edge.P_, key=lambda P: P.yx[0], reverse=True)  # sort Ps in descending order (bottom up)
            # scan to update link_:
            for i, P in enumerate(edge.P_):
                y, x = P.yx  # pivot, change to P center
                for _P in edge.P_[i+1:]:  # scan all higher Ps to get links to adjacent / overlapping Ps in P_ sorted by y
                    _y, _x = _P.yx
                    # get max possible y,x extension from P centers:
                    Dy = abs(P.yx_[0][0] - P.yx_[-1][0])/2; _Dy = abs(_P.yx_[0][0] - _P.yx_[-1][0])/2
                    Dx = abs(P.yx_[0][1] - P.yx_[-1][1])/2; _Dx = abs(_P.yx_[0][1] - _P.yx_[-1][1])/2
                    # min gap = distance between centers - combined extension,
                    # max overlap is negative min gap:
                    ygap = (_P.yx[0] - P.yx[0]) - (Dy+_Dy)
                    xgap = abs(_P.yx[1]-P.yx[1]) - (Dx+_Dx)
                    # overlapping | adjacent Ps:
                    if ygap <= 0 and xgap <= 0:
                        angle = np.subtract((y,x),(_y,_x))
                        P.link_[0] += [Clink(node=P, _node=_P, distance=np.hypot(*angle), angle=angle)]  # prelinks

        def select_max(edge):
            max_ = []
            for (y, x), (i, gy, gx, g) in edge.dert_.items():
                # sin_angle, cos_angle:
                sa, ca = gy/g, gx/g
                # get neighbor direction
                dy = 1 if sa > octant else -1 if sa < -octant else 0
                dx = 1 if ca > octant else -1 if ca < -octant else 0
                new_max = True
                for _y, _x in [(y-dy, x-dx), (y+dy, x+dx)]:
                    if (_y, _x) not in edge.dert_: continue  # skip if pixel not in edge blob
                    _i, _gy, _gx, _g = edge.dert_[_y, _x]  # get g of neighbor
                    if g < _g:
                        new_max = False
                        break
                if new_max: max_ += [((y, x), (sa, ca))]
            return max_

    CBlob = CEdge


class Clink(CBase):  # the product of comparison between two nodes

    def __init__(l,_node=None, node=None, dderH= None, roott=None, distance=0.0, angle=None):
        super().__init__()
        l.Et = [0,0,0,0,0,0]  # graph-specific, accumulated from surrounding nodes in node_connect
        l.node_ = []  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.link_ = []  # list of mediating Clinks in hyperlink
        l.dderH = CH() if dderH is None else dderH
        l.roott = [None, None] if roott is None else roott  # clusters that contain this link
        l.distance = distance  # distance between node centers
        l.angle = [0,0] if angle is None else angle  # dy,dx between node centers
        # dir: bool  # direction of comparison if not G0,G1, only needed for comp link?
        # deprecated:
        l._node = _node  # prior comparand
        l.node = node
        l.med_Gl_ = []  # replace by link_, intermediate nodes and links in roughly the same direction, as in hypergraph edges

    def __bool__(l): bool(l.dderH.H)
    # draft:
    def comp_link(_link, link, dderH, rn=1, fagg=0, flat=1):  # use in der+ and comp_kernel

        dderH = comp_(_link.dderH, link.dderH, dderH, rn=1, fagg=0, flat=1)
        mA,dA = comp_angle(_link.angle, link.angle)
        # draft:
        for _med_link,med_link in zip(_link.link_,link.link_):
            comp_link(_med_link, med_link)


class CP(CBase):
    def __init__(P, edge, yx, axis, dert):  # form_P:

        super().__init__()
        y, x = yx
        pivot = i, gy, gx, g = dert
        ma = ave_dangle  # max value because P direction is the same as dert gradient direction
        m = ave_g - g
        pivot += ma, m   # pack extra ders

        I, G, M, Ma, L, Dy, Dx = i, g, m, ma, 1, gy, gx
        P.axis = ay, ax = axis
        P.yx_, P.dert_, P.link_ = [yx], [pivot], [[]]

        for dy, dx in [(-ay, -ax), (ay, ax)]: # scan in 2 opposite directions to add derts to P
            P.yx_.reverse(); P.dert_.reverse()
            (_y, _x), (_, _gy, _gx, *_) = yx, pivot  # start from pivot
            y, x = _y+dy, _x+dx  # 1st extension
            while True:
                # scan to blob boundary or angle miss:
                try: i, gy, gx, g = interpolate2dert(edge, y, x)
                except TypeError: break  # out of bound (TypeError: cannot unpack None)

                mangle,dangle = comp_angle((_gy,_gx), (gy, gx))
                if abs(mangle*2-1) < ave_dangle: break  # terminate P if angle miss
                # update P:
                m = ave_g - g
                I += i; Dy += dy; Dx += dx; G += g; Ma += ma; M += m; L += 1
                P.yx_ += [(y, x)]; P.dert_ += [(i, gy, gx, g, ma, m)]
                # for next loop:
                y += dy; x += dx
                _y, _x, _gy, _gx = y, x, gy, gx

        P.yx = P.yx_[L // 2]
        P.latuple = I, G, M, Ma, L, (Dy, Dx)
        P.derH = CH()

    def __repr__(P): return f"P({', '.join(map(str, P.latuple))})"  # or return f"P(id={P.id})" ?

def interpolate2dert(edge, y, x):
    if (y, x) in edge.dert_: return edge.dert_[y, x]  # if edge has (y, x) in it

    # get nearby coords:
    y_ = [fy] = [floor(y)]; x_ = [fx] = [floor(x)]
    if y != fy: y_ += [fy+1]    # y is non-integer
    if x != fx: x_ += [fx+1]    # x is non-integer

    I, Dy, Dx, G = 0, 0, 0, 0
    for _y in y_:
        for _x in x_:
            if (_y, _x) not in edge.dert_: return
            i, dy, dx, g = edge.dert_[_y, _x]
            k = (1 - abs(_y-y)) * (1 - abs(_x-x))
            I += i*k; Dy += dy*k; Dx += dx*k; G += g*k

    return I, Dy, Dx, G

def comp_angle(_A, A):  # rn doesn't matter for angles

    _angle, angle = [atan2(Dy, Dx) for Dy, Dx in [_A, A]]
    dangle = _angle - angle  # difference between angles

    if dangle > pi: dangle -= 2*pi  # rotate full-circle clockwise
    elif dangle < -pi: dangle += 2*pi  # rotate full-circle counter-clockwise

    mangle = (cos(dangle)+1)/2  # angle similarity, scale to [0,1]
    dangle /= 2*pi  # scale to the range of mangle, signed: [-.5,.5]

    return [mangle, dangle]

def project(y, x, s, c, r):
    dist = s*y + c*x - r
    # Subtract left and right side by dist:
    # 0 = s*y + c*x - r - dist
    # 0 = s*y + c*x - r - dist*(s*s + c*c)
    # 0 = s*(y - dist*s) + c*(x - dist*c) - r
    # therefore, projection of y, x onto the line is:
    return y - dist*s, x - dist*c

if __name__ == "__main__":

    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)

    frame = CsliceEdge(image).segment()
    # verification:
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # # show first largest n edges
    # edge_, edgeQue = [], list(frame.blob_)
    # while edgeQue:
    #     blob = edgeQue.pop(0)
    #     if hasattr(blob, "P_"): edge_ += [blob]
    #     elif hasattr(blob, "rlay"): edgeQue += blob.rlay.blob_
    #
    # num_to_show = 5
    # sorted_edge_ = sorted(edge_, key=lambda edge: len(edge.yx_), reverse=True)
    # for edge in sorted_edge_[:num_to_show]:
    #     yx_ = np.array(edge.yx_)
    #     yx0 = yx_.min(axis=0) - 1
    #
    #     # show edge-blob
    #     shape = yx_.max(axis=0) - yx0 + 2
    #     mask_nonzero = tuple(zip(*(yx_ - yx0)))
    #     mask = np.zeros(shape, bool)
    #     mask[mask_nonzero] = True
    #     plt.imshow(mask, cmap='gray', alpha=0.5)
    #     plt.title(f"area = {edge.area}")
    #
    #     # show gradient
    #     vu_ = [(-gy/g, gx/g) for i, gy, gx, g in edge.dert_.values()]
    #     y_, x_ = zip(*(yx_ - yx0))
    #     v_, u_ = zip(*vu_)
    #     plt.quiver(x_, y_, u_, v_)
    #
    #     # show slices
    #     edge.P_.sort(key=lambda P: len(P.yx_), reverse=True)
    #     for P in edge.P_:
    #         yx1, yx2 = P.yx_[0], P.yx_[-1]
    #         y_, x_ = zip(*(P.yx_ - yx0))
    #         yp, xp = P.yx - yx0
    #         plt.plot(x_, y_, "g-", linewidth=2)
    #         for link in P.link_[-1]:
    #             _yp, _xp = link._node.yx - yx0
    #             plt.plot([_xp, xp], [_yp, yp], "ko-")
    #
    #         # # show slice axis
    #         # s, c = P.axis
    #         # vu_ = [(gy / g, gx / g) for i, gy, gx, g in edge.dert_.values()]
    #         # plt.quiver(xp, yp, s, c, color='b')
    #
    #     ax = plt.gca()
    #     ax.set_aspect('equal', adjustable='box')
    #     plt.show()