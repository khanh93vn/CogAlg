# warnings.filterwarnings('error')
# import warnings  # to detect overflow issue, in case of infinity loop
from itertools import zip_longest
import sys
import numpy as np
from copy import copy, deepcopy
from itertools import product
from .classes import Cptuple, CP, CPP, CderP
from .filters import ave, ave_g, ave_ga, ave_rotate
from .comp_slice import comp_slice, comp_angle
from .agg_convert import agg_recursion_eval
from .sub_recursion import sub_recursion_eval

'''
Vectorize is a terminal fork of intra_blob.
-
In natural images, objects look very fuzzy and frequently interrupted, only vaguely suggested by initial blobs and contours.
Potential object is proximate low-gradient (flat) blobs, with rough / thick boundary of adjacent high-gradient (edge) blobs.
These edge blobs can be dimensionality-reduced to their long axis / median line: an effective outline of adjacent flat blob.
-
Median line can be connected points that are most equidistant from other blob points, but we don't need to define it separately.
An edge is meaningful if blob slices orthogonal to median line form some sort of a pattern: match between slices along the line.
In simplified edge tracing we cross-compare among blob slices in x along y, where y is the longer dimension of a blob / segment.
Resulting patterns effectively vectorize representation: they represent match and change between slice parameters along the blob.
-
This process is very complex, so it must be selective. Selection should be by combined value of gradient deviation of edge blobs
and inverse gradient deviation of flat blobs. But the latter is implicit here: high-gradient areas are usually quite sparse.
A stable combination of a core flat blob with adjacent edge blobs is a potential object.
-
So, comp_slice traces blob axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)
-
Vectorization is clustering of parameterized Ps + their derivatives (derPs) into PPs: patterns of Ps that describe edge blob.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.
As we add higher dimensions (3D and time), this dimensionality reduction is done in salient high-aspect blobs
(likely edges in 2D or surfaces in 3D) to form more compressed "skeletal" representations of full-dimensional patterns.
'''

def vectorize_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    slice_blob(blob, verbose=verbose)  # form 2D array of Ps: horizontal blob slices in dert__
    rotate_P_(blob)  # re-form Ps around centers along P.G, P sides may overlap, if sum(P.M s + P.Ma s)?
    form_link_(blob)  # trace adjacent Ps, fill|prune if missing or redundant, add them to P.link_

    comp_slice(blob, verbose=verbose)  # scan rows top-down, compare y-adjacent, x-overlapping Ps to form derPs
    for fd, PP_ in enumerate([blob.PPm_, blob.PPd_]):
        sub_recursion_eval(blob, PP_, fd=fd)  # intra PP, no blob fb
        # compare PPs, cluster in graphs:
        if sum([PP.valt[fd] for PP in PP_]) > ave * sum([PP.rdnt[fd] for PP in PP_]):
            agg_recursion_eval(blob, copy(PP_), fd=fd)  # comp sub_PPs, form intermediate PPs

def form_link_(blob):  # trace adjacent Ps by adjacent dert roots, fill|prune if missing or redundant, add to P.link_ if >ave*rdn
    pass

'''
or only compute params needed for rotate_P_?
'''
def slice_blob(blob, verbose=False):  # form blob slices nearest to slice Ga: Ps, ~1D Ps, in select smooth edge (high G, low Ga) blobs

    mask__ = blob.mask__  # same as positive sign here
    dert__ = zip(*blob.dert__)  # convert 10-tuple of 2D arrays into 1D array of 10-tuple blob rows
    dert__ = [zip(*dert_) for dert_ in dert__]  # convert 1D array of 10-tuple rows into 2D array of 10-tuples per select blob
    blob.dert__ = dert__
    P__ = []
    height, width = mask__.shape
    if verbose: print("Converting to image...")

    for y, (dert_, mask_) in enumerate(zip(dert__, mask__)):  # unpack lines, each may have multiple slices -> Ps:
        if verbose: print(f"\rConverting to image... Processing line {y + 1}/{height}", end=""); sys.stdout.flush()
        P_ = []
        _mask = True  # mask -1st dert
        x = 0
        for (i, *dert), mask in zip(dert_, mask_):
            g, ga, ri, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = dert
            if not mask:  # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination
                if _mask:  # ini P params with first unmasked dert
                    Pdert_ = [dert]
                    I = ri; M = ave_g - g; Ma = ave_ga - ga; Dy = dy; Dx = dx
                    Sin_da0, Cos_da0, Sin_da1, Cos_da1 = sin_da0, cos_da0, sin_da1, cos_da1
                else:
                    # dert and _dert are not masked, accumulate P params:
                    I +=ri; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx  # angle
                    Sin_da0+=sin_da0; Cos_da0+=cos_da0; Sin_da1+=sin_da1; Cos_da1+=cos_da1  # aangle
                    Pdert_ += [dert]
            elif not _mask:
                # _dert is not masked, dert is masked, terminate P:
                G = np.hypot(Dy, Dx)  # Dy,Dx  # recompute G,Ga, it can't reconstruct M,Ma
                Ga = (Cos_da0 + 1) + (Cos_da1 + 1)  # Cos_da0, Cos_da1
                L = len(Pdert_)  # params.valt = [params.M+params.Ma, params.G+params.Ga]?
                P_ += [Dert2P(I, M, Ma, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1, G, Ga, L, y, x, Pdert_, blob.root__)]
            _mask = mask
            x += 1
        # pack last P, same as above:
        if not _mask:
            G = np.hypot(Dy, Dx); Ga = (Cos_da0 + 1) + (Cos_da1 + 1)
            L = len(Pdert_) # params.valt=[params.M+params.Ma,params.G+params.Ga]
            P_ += [Dert2P(I, M, Ma, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1, G, Ga, L, y, x, Pdert_, blob.root__)]
        P__ += [P_]

    if verbose: print("\r", end="")
    blob.P__ = P__
    return P__

def Dert2P(I, M, Ma, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1, G, Ga, L, y, x, Pdert_, root__):

    P = CP(ptuple=[I, M, Ma, [Dy, Dx], [Sin_da0, Cos_da0, Sin_da1, Cos_da1], G, Ga, L], box=[y, y, x-L, x-1], dert_=Pdert_)
    P.dert_roots_ = [[P] for dert in Pdert_]

    bx = P.box[2]
    while bx < P.box[3]:  # x0->xn
        root__[y][bx] += [P]
        bx += 1

    return P

def rotate_P_(blob):  # rotate each P to align it with direction of P gradient

    P__, dert__, mask__ = blob.P__, blob.dert__, blob.mask__

    for P_ in P__:
        for P in P_:
            daxis = P.ptuple[3][0] / P.ptuple[5]  # dy: deviation from horizontal axis
            _daxis = 0
            G = P.ptuple[5]
            while abs(daxis)*G > ave_rotate:  # recursive reform P along new G angle in blob.dert__, P.daxis for future reval?

                rotate_P(P, dert__, mask__, ave_a=None)  # rescan in the direction of ave_a, if any
                maxis, daxis = comp_angle(P.ptuple[3], P.axis)
                ddaxis = daxis +_daxis  # cancel-out if opposite-sign
                # terminate if oscillation
                if ddaxis*G < ave_rotate:
                    rotate_P(P, dert__, mask__, ave_a=np.add(P.ptuple[3], P.axis))  # rescan in the direction of ave_a, if any
                    break

def rotate_P(P, dert__, mask__, ave_a):

    if ave_a is None:
        sin, cos = np.divide(P.ptuple[3], P.ptuple[5])
    else:
        sin, cos = np.divide(ave_a, np.hypot(*ave_a))
    new_axis = sin, cos

    if cos < 0: sin,cos = -sin,-cos  # dx always >= 0, dy can be < 0
    y0,yn,x0,xn = P.box
    ycenter = (y0+yn)/2; xcenter = (x0+xn)/2
    rdert_ = []
    # scan left:
    rx=xcenter; ry=ycenter
    while True:  # terminating condition is in form_rdert()
        rdert = form_rdert(rx,ry, dert__, mask__)
        if rdert is None: break  # dert is not in blob: masked or out of bound
        rdert_ = [rdert] + rdert_  # append left
        rx-=cos; ry-=sin  # next rx,ry
    x0 = rx; yleft = ry
    # scan right:
    rx=xcenter+cos; ry=ycenter+sin  # center dert was included in scan left
    while True:
        rdert = form_rdert(rx,ry, dert__, mask__)
        if rdert is None: break  # dert is not in blob: masked or out of bound
        rdert_ += [rdert]  # append right
        rx+=cos; ry+=sin  # next rx,ry
    # form rP:
    if not rdert_: return
    rdert = rdert_[0]  # initialization:
    G, Ga, I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1 = rdert; M=ave_g-G; Ma=ave_ga-Ga; dert_=[rdert]
    # accumulation:
    for rdert in rdert_[1:]:
        g, ga, i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = rdert
        I+=i; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx; Sin_da0+=sin_da0; Cos_da0+=cos_da0; Sin_da1+=sin_da1; Cos_da1+=cos_da1
        dert_ += [rdert]
    # re-form gradients:
    G = np.hypot(Dy,Dx);  Ga = (Cos_da0 + 1) + (Cos_da1 + 1); L = len(rdert_)
    # replace P:
    P.ptuple = [I, M, Ma, [Dy, Dx], [Sin_da0, Cos_da0, Sin_da1, Cos_da1], G, Ga, L]
    P.dert_ = dert_
    P.box = [min(yleft, ry), max(yleft, ry), x0, rx]  # P may go up-right or down-right
    P.axis = new_axis

def form_rdert(rx,ry, dert__, mask__):

    Y, X = mask__.shape
    # coord, distance of four int-coord derts, overlaid by float-coord rdert in dert__, int for indexing
    # always in dert__ for intermediate float rx,ry:
    x1 = int(np.floor(rx)); dx1 = abs(rx - x1)
    x2 = int(np.ceil(rx));  dx2 = abs(rx - x2)
    y1 = int(np.floor(ry)); dy1 = abs(ry - y1)
    y2 = int(np.ceil(ry));  dy2 = abs(ry - y2)
    # terminate scan_left | scan_right:
    if (x1 < 0 or x1 >= X or x2 < 0 or x2 >= X) or (y1 < 0 or y1 >= Y or y2 < 0 or y2 >= Y):
        return None
    # scale all dert params in proportion to inverted distance from rdert, sum(distances) = 1
    # approximation, square of rpixel is rotated, won't fully match not-rotated derts
    k0 = 2 - dx1*dx1 - dy1*dy1
    k1 = 2 - dx1*dx1 - dy2*dy2
    k2 = 2 - dx2*dx2 - dy1*dy1
    k3 = 2 - dx2*dx2 - dy2*dy2
    K = k0 + k1 + k2 + k3
    mask = (
        mask__[y1, x1] * k0 +
        mask__[y2, x1] * k1 +
        mask__[y1, x2] * k2 +
        mask__[y2, x2] * k3
           ) / K
    if round(mask):  # summed mask is fractional, round to 1|0
        return None  # return rdert if inside the blob

    ptuple = []
    for par0, par1, par2, par3 in (zip(dert__[y1,x1][1:], dert__[y2,x1][1:], dert__[y1,x2][1:], dert__[y2,x2][1:])):  # skip i
        ptuple += [(par0*k0 + par1*k1 + par2*k2 + par3*k3) / K]

    return ptuple


def slice_blob_ortho(blob, verbose=False):  # slice_blob with axis-orthogonal Ps

    from .hough_P import new_rt_olp_array, hough_check
    Y, X = blob.mask__.shape
    # Get thetas and positions:
    dy__, dx__ = blob.dert__[4:6]  # Get blob derts' angle
    y__, x__ = np.indices((Y, X))  # Get blob derts' position
    theta__ = np.arctan2(dy__, dx__)  # Compute theta

    if verbose:
        step = 100 / (~blob.mask__).sum()  # progress % percent per pixel
        progress = 0.0; print(f"\rFilling... {round(progress)} %", end="");  sys.stdout.flush()
    # derts with same rho and theta lies on the same line
    # floodfill derts with similar rho and theta
    P_ = []
    filled = blob.mask__.copy()
    for y in y__[:, 0]:
        for x in x__[0]:
            # initialize P at first unfilled dert found
            if not filled[y, x]:
                M = 0; Ma = 0; I = 0; Dy = 0; Dx = 0; Sin_da0 = 0; Cos_da0 = 0; Sin_da1 = 0; Cos_da1 = 0
                dert_ = []
                box = [y, y, x, x]
                to_fill = [(y, x)]                  # dert indices to floodfill
                rt_olp__ = new_rt_olp_array((Y, X)) # overlap of rho theta (line) space
                while to_fill:                      # floodfill for one P
                    y2, x2 = to_fill.pop()          # get next dert index to fill
                    if x2 < 0 or x2 >= X or y2 < 0 or y2 >= Y:  # skip if out of bounds
                        continue
                    if filled[y2, x2]:              # skip if already filled
                        continue
                    # check if dert is almost on the same line and have similar gradient angle
                    new_rt_olp__ = hough_check(rt_olp__, y2, x2, theta__[y2, x2])
                    if not new_rt_olp__.any():
                        continue

                    filled[y2, x2] = True       # mark as filled
                    rt_olp__[:] = new_rt_olp__  # update overlap
                    # accumulate P params:
                    dert = tuple(param__[y2, x2] for param__ in blob.dert__[1:])
                    g, ga, ri, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = dert  # skip i
                    M += ave_g - g; Ma += ave_ga - ga; I += ri; Dy += dy; Dx += dx
                    Sin_da0 += sin_da0; Cos_da0 += cos_da0; Sin_da1 += sin_da1; Cos_da1 += cos_da1
                    dert_ += [(y2, x2, *dert)]  # unpack x, y, add dert to P

                    if y2 < box[0]: box[0] = y2
                    if y2 > box[1]: box[1] = y2
                    if x2 < box[2]: box[2] = x2
                    if x2 > box[3]: box[3] = x2
                    # add neighbors to fill
                    to_fill += [*product(range(y2-1, y2+2), range(x2-1, x2+2))]
                if not rt_olp__.any():
                    raise ValueError
                G = np.hypot(Dy, Dx)  # Dy,Dx  # recompute G,Ga, it can't reconstruct M,Ma
                Ga = (Cos_da0 + 1) + (Cos_da1 + 1)  # Cos_da0, Cos_da1
                L = len(dert_)
                if G == 0:
                    axis = 0, 1
                else:
                    axis = Dy / G, Dx / G

                P_ += [CP(ptuple=[I, M, Ma, [Dy, Dx], [Sin_da0, Cos_da0, Sin_da1, Cos_da1], G, Ga, L],
                          box=box, dert_=dert_, axis=axis)]
                if verbose:
                    progress += L * step; print(f"\rFilling... {round(progress)} %", end=""); sys.stdout.flush()
    blob.P__ = [P_]
    if verbose: print("\r" + " " * 79, end=""); sys.stdout.flush(); print("\r", end="")
    return P_


def slice_blob_flow(blob, verbose=False):  # version of slice_blob_ortho

    # find the derts with gradient pointing at current dert:
    _yx_ = np.indices(blob.mask__.shape)[:, ~blob.mask__].T  # blob derts' position
    with np.errstate(divide='ignore', invalid='ignore'):  # suppress numpy RuntimeWarning
        sc_ = np.divide(blob.dert__[4:6], blob.dert__[1])[:, ~blob.mask__].T    # blob derts' angle
    uv_ = np.zeros_like(sc_)        # (u, v) points to one of the eight neighbor cells
    u_, v_ = uv_.T                  # unpack u, v
    s_, c_ = sc_.T                  # unpack sin, cos
    u_[0.5 <= s_] = 1              # down, down left or down right
    u_[(-0.5 < s_) & (s_ < 0.5)] = 0  # left or right
    u_[s_ <= -0.5] = -1              # up, up-left or up-right
    v_[0.5 <= c_] = 1              # right, up-right or down-right
    v_[(-0.5 < c_) & (c_ < 0.5)] = 0  # up or down
    v_[c_ <= -0.5] = -1              # left, up-left or down-left
    yx_ = _yx_ + uv_                # compute target cell position
    m__ = (yx_.reshape(-1, 1, 2) == _yx_).all(axis=2)   # mapping from _yx_ to yx_
    def get_p(a):
        nz = a.nonzero()[0]
        if len(nz) == 0:    return -1
        elif len(nz) == 1:  return nz[0]
        else:               raise ValueError
    p_ = [*map(get_p, m__)]       # reduced mapping from _yx_ to yx_
    n_ = m__.sum(axis=0) # find n, number of gradient sources per cell

    # cluster Ps, start from cells without any gradient source
    P_ = []
    for i in range(len(n_)):
        if n_[i] == 0:                  # start from cell without any gradient source
            I = 0; M = 0; Ma = 0; Dy = 0; Dx = 0; Sin_da0 = 0; Cos_da0 = 0; Sin_da1 = 0; Cos_da1 = 0
            dert_ = []
            y, x = _yx_[i]
            box = [y, y, x, x]

            j = i
            while True:      # while there is a dert to follow
                y, x = _yx_[j]      # get dert position
                dert = [par__[y, x] for par__ in blob.dert__[1:]]  # dert params at _y, _x, skip i
                g, ga, ri, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = dert
                I+=i; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx; Sin_da0+=sin_da0; Cos_da0+=cos_da0; Sin_da1+=sin_da1; Cos_da1+=cos_da1
                dert_ += [(y, x, *dert)]
                if y < box[0]: box[0] = y
                if y > box[1]: box[1] = y
                if x < box[2]: box[2] = x
                if x > box[3]: box[3] = x

                # remove all gradient sources from the cell
                while True:
                    try:
                        k = p_.index(j)
                        p_[k] = -1
                    except ValueError as e:
                        if "is not in list" not in str(e):
                            raise e
                        break
                if p_[j] != -1:
                    j = p_[j]
                else:
                    break
            G = np.hypot(Dy, Dx); Ga = (Cos_da0 + 1) + (Cos_da1 + 1)
            L = len(dert_) # params.valt=[params.M+params.Ma,params.G+params.Ga]
            P_ += [CP(ptuple=[I,M,Ma,[Dy,Dx],[Sin_da0,Cos_da0,Sin_da1,Cos_da1], G, Ga, L], box=[y,y, x-L,x-1], dert_=dert_)]

    blob.P__ = [P_]

    return blob.P__

def append_P(P__, P):  # pack P into P__ in top down sequence

    current_ys = [P_[0].y0 for P_ in P__]  # list of current-layer seg rows
    if P.y0 in current_ys:
        if P not in P__[current_ys.index(P.y0)]:
            P__[current_ys.index(P.y0)].append(P)  # append P row
    elif P.y0 > current_ys[0]:  # P.y0 > largest y in ys
        P__.insert(0, [P])
    elif P.y0 < current_ys[-1]:  # P.y0 < smallest y in ys
        P__.append([P])
    elif P.y0 < current_ys[0] and P.y0 > current_ys[-1]:  # P.y0 in between largest and smallest value
        for i, y in enumerate(current_ys):  # insert y if > next y
            if P.y0 > y: P__.insert(i, [P])  # PP.P__.insert(P.y0 - current_ys[-1], [P])


def copy_P(P, Ptype=None):  # Ptype =0: P is CP | =1: P is CderP | =2: P is CPP | =3: P is CderPP | =4: P is CaggPP

    if not Ptype:  # assign Ptype based on instance type if no input type is provided
        if isinstance(P, CPP):     Ptype = 2
        elif isinstance(P, CderP): Ptype = 1
        else:                      Ptype = 0  # CP

    uplink_layers, downlink_layers = P.uplink_layers, P.downlink_layers  # local copy of link layers
    P.uplink_layers, P.downlink_layers = [], []  # reset link layers
    roott = P.roott  # local copy
    P.roott = [None, None]
    if Ptype == 1:
        P_derP, _P_derP = P.P, P._P  # local copy of derP.P and derP._P
        P.P, P._P = None, None  # reset
    elif Ptype == 2:
        mseg_levels, dseg_levels = P.mseg_levels, P.dseg_levels
        P__ = P.P__
        P.mseg_levels, P.dseg_levels, P.P__ = [], [], []  # reset
    elif Ptype == 3:
        PP_derP, _PP_derP = P.PP, P._PP  # local copy of derP.P and derP._P
        P.PP, P._PP = None, None  # reset
    elif Ptype == 4:
        gPP_, cPP_ = P.gPP_, P.cPP_
        mlevels, dlevels = P.mlevels, P.dlevels
        P.gPP_, P.cPP_, P, P.mlevels, P.dlevels = [], [], [], []  # reset

    new_P = deepcopy(P)  # copy P with empty root and link layers, reassign link layers:
    new_P.uplink_layers += copy(uplink_layers)
    new_P.downlink_layers += copy(downlink_layers)

    # shallow copy to create new list
    P.uplink_layers, P.downlink_layers = copy(uplink_layers), copy(downlink_layers)  # reassign link layers
    P.roott = roott  # reassign root
    # reassign other list params
    if Ptype == 1:
        new_P.P, new_P._P = P_derP, _P_derP
        P.P, P._P = P_derP, _P_derP
    elif Ptype == 2:
        P.mseg_levels, P.dseg_levels = mseg_levels, dseg_levels
        new_P.P__ = copy(P__)
        new_P.mseg_levels, new_P.dseg_levels = copy(mseg_levels), copy(dseg_levels)
    elif Ptype == 3:
        new_P.PP, new_P._PP = PP_derP, _PP_derP
        P.PP, P._PP = PP_derP, _PP_derP
    elif Ptype == 4:
        P.gPP_, P.cPP_ = gPP_, cPP_
        P.roott = roott
        new_P.gPP_, new_P.cPP_ = [], []
        new_P.mlevels, new_P.dlevels = copy(mlevels), copy(dlevels)

    return new_P