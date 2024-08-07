import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy, copy
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
import math as math
from comp_slice import *
'''
Blob edges may be represented by higher-composition PPPs, etc., if top param-layer match,
in combination with spliced lower-composition PPs, etc, if only lower param-layers match.
This may form closed edge patterns around flat blobs, which defines stable objects.   
'''

# agg-recursive versions should be more complex?
class CderPP(ClusterStructure):  # tuple of derivatives in PP uplink_ or downlink_, PP can also be PPP, etc.

    # draft
    params = list  # PP derivation layer, flat, decoded by mapping each m,d to lower-layer param
    x0 = int  # redundant to params:
    x = float  # median x
    L = int  # pack in params?
    sign = NoneType  # g-ave + ave-ga sign
    y = int  # for vertical gaps in PP.P__, replace with derP.P.y?
    PP = object  # lower comparand
    _PP = object  # higher comparand
    root = lambda:None  # segment in sub_recursion
    # higher derivatives
    rdn = int  # mrdn, + uprdn if branch overlap?
    uplink_layers = lambda: [[],[]]  # init a layer of dderPs and a layer of match_dderPs
    downlink_layers = lambda: [[],[]]
   # from comp_dx
    fdx = NoneType

class CPPP(CPP, CderPP):

    # draft
    params = list  # derivation layers += derP params per der+, param L is actually Area
    sign = bool
    rng = lambda: 1  # rng starts with 1
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only
    nP = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    uplink_layers = lambda: [[],[]]
    downlink_layers = lambda: [[],[]]
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list  # for visualization only, original box before flipping
    mask__ = bool
    P__ = list  # input  # derP__ = list  # redundant to P__
    seg_levels = lambda: [[[]],[[]]]  # from 1st agg_recursion, seg_levels[0] is seg_t, higher seg_levels are segP_t s
    PPP_levels = list  # from 2nd agg_recursion, PP_t = levels[0], from form_PP, before recursion
    layers = list  # from sub_recursion, each is derP_t
    root = lambda:None  # higher-order segP or PPP


def agg_recursion(blob, fseg):  # compositional recursion per blob.Plevel. P, PP, PPP are relative terms, each may be of any composition order

    if fseg: PP_t = [blob.seg_levels[0][-1], blob.seg_levels[1][-1]]   # blob is actually PP, recursion forms segP_t, seg_PP_t, etc.
    else:    PP_t = [blob.levels[0][-1], blob.levels[1][-1]]  # input-level composition Ps, initially PPs

    PPP_t = []  # next-level composition Ps, initially PPPs  # for fiPd, PP_ in enumerate(PP_t): fiPd = fiPd % 2  # dir_blob.M += PP.M += derP.m
    n_extended = 0

    for i, PP_ in enumerate(PP_t):   # fiPd = fiPd % 2
        fiPd = i % 2
        if fiPd: ave_PP = ave_dPP
        else:    ave_PP = ave_mPP
        if fseg: M = ave- blob.params[-1][fiPd][4]  # blob.params[0][fiPd][4] is mG | dG
        else: M = ave-abs(blob.G)  # if M > ave_PP * blob.rdn and len(PP_)>1:  # >=2 comparands

        if len(PP_)>1:
            n_extended += 1
            derPP_t = comp_PP_(PP_)  # compare all PPs to the average (centroid) of all other PPs, is generic for lower level
            PPP_t = form_PPP_t(derPP_t)
            # call individual comp_PP if mPPP > ave_mPPP, converting derPP to CPPP
            splice_PPs(PPP_t)  # for initial PPs only: if PP is CPP?
            sub_recursion_eval(PPP_t)  # rng+ or der+, if PP is CPPP?
        else:
            PPP_t += [[], []]  # replace with neg PPPs?

    if fseg: blob.seg_levels += [PPP_t]  # new level of segPs
    else:    blob.levels += [PPP_t]  # levels of dir_blob are Plevels

    if n_extended/len(PP_t) > 0.5:  # mean ratio of extended PPs
        agg_recursion(blob, fseg)
'''
- Compare each PP to the average (centroid) of all other PPs in PP_, or maximal cartesian distance, forming derPPs.  
- Select above-average derPPs as PPPs, representing summed derivatives over comp range, overlapping between PPPs.
'''

def comp_PP_(PP_):  # PP can also be PPP, etc.

    derPPm_, derPPd_ = [],[]

    for PP in PP_:
        compared_PP_ = copy(PP_)  # shallow copy
        compared_PP_.remove(PP)
        n = len(compared_PP_)
        # sum same-type params across compared PPs:
        summed_params = deepcopy(compared_PP_[0].params)  # init 1st sum params with 1st element
        for compared_PP in compared_PP_[1:]:  # sum starts with 2nd element
            sum_nested_layer(summed_params, compared_PP.params)  # use generic unpack function?
        # ave params of compared PP:
        ave_params = get_layers_average(summed_params, n)  # use generic unpack function?

        derPP = CPP(params=deepcopy(PP.params), layers=[PP_])  # derPP inherits PP.params
        '''
        comp to ave params of compared PPs, form new layer: derivatives of all lower layers, 
        initial 3 layer nesting diagram: https://github.com/assets/52521979/ea6d436a-6c5e-429f-a152-ec89e715ebd6
        '''
        for i, (_layer, layer) in enumerate( zip(PP.params, ave_params)):
            derPP.params += [comp_layer(_layer, layer, i, der_layer=[])]  # recursive layer unpack to the depth=i

        derPPm_.append(copy_P(derPP, Ptype=2))
        derPPd_.append(copy_P(derPP, Ptype=2))

    return derPPm_, derPPd_


def form_PPP_t(derPP_t):  # form PPs from match-connected segs
    PPP_t = []

    for fPd, derPP_ in enumerate(derPP_t):
        # sort by value of last layer: derivatives of all lower layers:
        derPP_ = sorted(derPP_, key=lambda derPP: derPP.params[-1][fPd], reverse=True)  # descending order
        PPP_ = []
        for i, derPP in enumerate(derPP_):
            derPP_val = 0
            for param_layer in derPP.params:  # may need recursive unpack here
                derPP.rdn += param_layer[fPd] > param_layer[1-fPd]
                derPP_val += param_layer[fPd]  # make it a param?

            ave = vaves[fPd] * derPP.rdn * (i+1)  # derPP is redundant to higher-value previous derPPs in derPP_
            if derPP_val > ave:
                PPP_ += [derPP]  # base derPP and PPP is CPP
                if derPP_val > ave*10:
                    ind_comp_PP_(derPP, fPd)  # derPP is converted from CPP to CPPP
            else:
                break  # ignore below-ave PPs
        PPP_t.append(PPP_)
    return PPP_t

# draft
def ind_comp_PP_(_PP, fPd):  # 1-to-1 comp, _PP is converted from CPP to higher-composition CPPP

    derPP_ = []
    rng = _PP.params[-1][fPd] / 3  # 3: ave per rel_rng+=1, actual rng is Euclidean distance:

    for PP in _PP.layers[0]:  # 1-to-1 comparison between _PP and other PPs within rng
        derPP = CderPP()
        _area = _PP.params.L  # pseudo, we need L index in params
        area = PP.params.L
        dx = _PP.x/_area - PP.x/area
        dy = _PP.y/_area - PP.y/area
        distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids
        _val = _PP.params[-1][fPd]
        val = PP.params[-1][fPd]
        if distance / ((_val+val)/2) < rng:  # distance relative to value, vs. area?

            derPP.params += [comp_params(PP.params[0], PP.params[0])]  # reform to compare 2tuples
            for i, _layer, layer in enumerate(zip(PP.params[1:], PP.params[1:])):
                derPP.params += [comp_layer(_layer, layer, i, der_layer=[])]  # recursive layer unpack to the depth=i
            derPP_ += [derPP]


    for i, _derPP in enumerate(derPP_):  # cluster derPPs into PPPs by connectivity, overwrite derPP[i]

        if _derPP.params[-1][fPd]:
            PPP = CPPP(params=deepcopy(_derPP.params), layers=[_derPP.PP])
            PPP.accum_from(_derPP)  # initialization
            _derPP.root = PPP
            for derPP in derPP_[i+1:]:
                if not derPP.PP.root:
                    if derPP.params[-1][fPd]:  # positive and not in PPP yet
                        PPP.layers.append(derPP)  # multiple composition orders
                        PPP.accum_from(_derPP)
                        derPP.root = PPP
                    # pseudo:
                    elif sum([derPP.params[:-1]][fPd]) > ave*len(derPP.params)-1:
                         # splice PP and their segs
                         pass
    '''
    if derPP.match params[-1]: form PPP
    elif derPP.match params[:-1]: splice PPs and their segs? 
    '''

# draft:
def comp_layer(_layer, layer, i, der_layer):  # nlists = max_nesting = i-1

    i -= 1  # nlists and max_nesting in sub_layer
    if i > 0:
        # keep unpacking
        for i, (_sub_layer, sub_layer) in enumerate( zip(_layer, layer)):  # sub_layers is a shorter version of layers
            der_layer += [comp_layer(_sub_layer, sub_layer, i, der_layer)]
    else:
        # nesting depth == 0; nlists in der_layer = sum(nlists in lower layers) * 2: 1, 2, 6, 18...
        der_layer += [comp_params(_layer, layer)]  # returns 2-tuple

    return der_layer

# unpack and and accum same-type params
# use the same unpack sequence as in comp_layer?
def sum_nested_layer(sum_layer, params_layer):

    if isinstance(sum_layer[0], list):  # if nested, continue to loop and search for deeper list
        for j, (sub_sum_layer, sub_params_layer) in enumerate(zip(sum_layer, params_layer)):
            sum_nested_layer(sub_sum_layer, sub_params_layer)
    else:  # if layer is not nested, sum params
        for j, param in enumerate(params_layer):
            sum_layer[j] += param

# pending update
# get average value for each param according to n value
def get_layers_average(sum_params, n):

    average_params = deepcopy(sum_params)  # get a copy as output
    if isinstance(average_params[0], list):  # if nested, continue to loop and search for deeper list
        for j, sub_sum_layer in enumerate(average_params):
            get_layers_average(sub_sum_layer, n)
    else:  # if layer is not nested, get average of each value
        for j, param in enumerate(average_params):
            average_params[j] = param/n

    return average_params


def comp_nested_layer(_param_layer, param_layer):

    if isinstance(_param_layer[0], list):   # if nested, continue to loop and search for deeper list
        sub_ders = []
        for j, (_sub_layer, sub_layer) in enumerate( zip(_param_layer, param_layer)):
            sub_ders += [comp_nested_layer(_sub_layer, sub_layer)]
        return sub_ders
    else:  # comp params if layer is not nested
        params, _, _ = comp_params(_param_layer, param_layer, nparams=len(_param_layer))
        mparams = params[0::2]  # get even index m params
        dparams = params[1::2]  # get odd index d params
        return [mparams, dparams]

# old:

def form_segPPP_root(PP_, root_rdn, fPd):  # not sure about form_seg_root

    for PP in PP_:
        link_eval(PP.uplink_layers, fPd)
        link_eval(PP.downlink_layers, fPd)

    for PP in PP_:
        form_segPPP_(PP)

def form_segPPP_(PP):
    pass

# pending update
def splice_segs(seg_):  # in 1st run of agg_recursion
    pass

# draft, splice 2 PPs for now
def splice_PPs(PP_, frng):  # splice select PP pairs if der+ or triplets if rng+

    spliced_PP_ = []
    while PP_:
        _PP = PP_.pop(0)  # pop PP, so that we can differentiate between tested and untested PPs
        tested_segs = []  # we need this because we may add new seg during splicing process, and those new seg need to check their link for splicing too
        _segs = _PP.seg_levels[0]

        while _segs:
            _seg = _segs.pop(0)
            _avg_y = sum([P.y for P in _seg.P__])/len(_seg.P__)  # y centroid for _seg

            for link in _seg.uplink_layers[1] + _seg.downlink_layers[1]:
                seg = link.P.root  # missing link of current seg

                if seg.root is not _PP:  # this may occur after the merging where multiple links are having same PP
                    avg_y = sum([P.y for P in seg.P__])/len(seg.P__)  # y centroid for seg

                    # test for y distance (temporary)
                    if (_avg_y - avg_y) < ave_splice:
                        if seg.root in PP_: PP_.remove(seg.root)  # remove merged PP
                        elif seg.root in spliced_PP_: spliced_PP_.remove(seg.root)
                        # splice _seg's PP with seg's PP
                        merge_PP(_PP, seg.root)

            tested_segs += [_seg]  # pack tested _seg
        _PP.seg_levels[0] = tested_segs
        spliced_PP_ += [_PP]

    return spliced_PP_

# to be updated
def merge_PP(_PP, PP, fPd):  # only for PP splicing

    for seg in PP.seg_levels[fPd][-1]:  # merge PP_segs into _PP:
        accum_CPP(_PP, seg, fPd)
        _PP.seg_levels[fPd][-1] += [seg]

    # merge uplinks and downlinks
    for uplink in PP.uplink_layers:
        if uplink not in _PP.uplink_layers:
            _PP.uplink_layers += [uplink]
    for downlink in PP.downlink_layers:
        if downlink not in _PP.downlink_layers:
            _PP.downlink_layers += [downlink]


def comp_dx(P):  # cross-comp of dx s in P.dert_

    Ddx = 0
    Mdx = 0
    dxdert_ = []
    _dx = P.dert_[0][2]  # first dx
    for dert in P.dert_[1:]:
        dx = dert[2]
        ddx = dx - _dx
        if dx > 0 == _dx > 0: mdx = min(dx, _dx)
        else: mdx = -min(abs(dx), abs(_dx))
        dxdert_.append((ddx, mdx))  # no dx: already in dert_
        Ddx += ddx  # P-wide cross-sign, P.L is too short to form sub_Ps
        Mdx += mdx
        _dx = dx
    P.dxdert_ = dxdert_
    P.Ddx = Ddx
    P.Mdx = Mdx

# june 22
def sub_recursion(root_layers, PP_, frng):  # compares param_layers of derPs in generic PP, form or accum top derivatives

    comb_layers = []
    for PP in PP_:  # PP is generic higher-composition pattern, P is generic lower-composition pattern
                    # both P and PP may be recursively formed higher-derivation derP and derPP, etc.
        if frng: PP_V = PP.params[-1][0] - ave_mPP * PP.rdn; rng = PP.rng+1; min_L = rng * 2  # V: value of sub_recursion per PP
        else:    PP_V = PP.params[-1][1] - ave_dPP * PP.rdn; rng = PP.rng; min_L = 3  # need 3 Ps to compute layer2, etc.
        if PP_V > 0 and PP.nderP > min_L:

            PP.rdn += 1  # rdn to prior derivation layers
            PP.rng = rng
            Pm__ = comp_P_rng(PP.P__, rng)
            Pd__ = comp_P_der(PP.P__)

            sub_segm_ = form_seg_root([Pm_ for Pm_ in reversed(Pm__)], root_rdn=PP.rdn, fPd=0)
            sub_segd_ = form_seg_root([Pd_ for Pd_ in reversed(Pd__)], root_rdn=PP.rdn, fPd=1)
            sub_PPm_, sub_PPd_ = form_PP_root(( sub_segm_, sub_segd_), base_rdn=PP.rdn)  # forms PPs: parameterized graphs of linked segs

            PP.layers = [(sub_PPm_, sub_PPd_)]
            if sub_PPm_:
                # rng+=1, |+=n to reduce clustering costs?
                sub_recursion(PP.layers, sub_PPm_, frng=1)  # rng+ comp_P in PPms, form param_layer, sub_PPs
            if sub_PPd_:
                sub_recursion(PP.layers, sub_PPd_, frng=0)  # der+ comp_P in PPds, form param_layer, sub_PPs

            if PP.layers:  # pack added sublayers:
                new_comb_layers = []
                for (comb_sub_PPm_, comb_sub_PPd_), (sub_PPm_, sub_PPd_) in zip_longest(comb_layers, PP.layers, fillvalue=([], [])):
                    comb_sub_PPm_ += sub_PPm_
                    comb_sub_PPd_ += sub_PPd_
                    new_comb_layers.append((comb_sub_PPm_, comb_sub_PPd_))  # add sublayer
                comb_layers = new_comb_layers

    if comb_layers: root_layers += comb_layers


def comp_P_rng(iP__, rng):  # rng+ sub_recursion in PP.P__, adding two link_layers per P

    P__ = [P_ for P_ in reversed(iP__)]  # revert to top-down
    uplinks__ = [[ [] for P in P_] for P_ in P__[rng:]]  # rng derP_s per P, exclude 1st rng rows
    downlinks__ = [[ [] for P in P_] for P_ in P__[:-rng]]  # exclude last rng rows

    for y, _P_ in enumerate(P__[:-rng]):  # higher compared row, skip last rng: no lower comparand rows
        for x, _P in enumerate(_P_):

            for pri_rng_derP in _P.downlink_layers[-1]:  # get linked Ps at dy = rng-1
                pri_P = pri_rng_derP.P
                for ini_derP in pri_P.downlink_layers[0]:  # lower comparands are linked Ps at dy = rng
                    P = ini_derP.P
                    if isinstance(P, CPP) or isinstance(P, CderP):  # rng+ fork for derPs, very unlikely
                        derP = comp_derP(_P, P)  # form higher vertical derivatives of derP or PP params
                    else:
                        derP = comp_P(_P, P)  # form vertical derivatives of horizontal P params
                    # += links:
                    downlinks__[y][x] += [derP]
                    up_x = P__[y+rng].index(P)  # index of P in P_ at y+rng
                    uplinks__[y][up_x] += [derP]  # uplinks__[y] = P__[y+rng]: uplinks__= P__[rng:]

    for P_, uplinks_ in zip( P__[rng:], uplinks__):  # skip 1st rmg rows, no uplinks
        for P, uplinks in zip(P_, uplinks_):
            P.uplink_layers += [uplinks, []]  # add rng_derP_ to P.link_layers

    for P_, downlinks_ in zip(P__[:-rng], downlinks__):  # skip last rng rows, no downlinks
        for P, downlinks in zip(P_, downlinks_):
            P.downlink_layers += [downlinks, []]

    return iP__  # return bottom-up P__


# replace with inline derP initialization?
def comp_derP(_derP, derP, instance=CderP, finP=1, foutderP=1):
    # instance, finP, foutderP are not needed anymore?

    derivatives_t = []
    mP = 0  # for rng+ eval
    dP = 0  # for der+ eval

    if finP:
        if isinstance(_derP, CderP):  # params is in tuple of 2, each with 10 elements
            derivatives_t = comp_ptuple(_derP.params, derP.params)

        else:  # params is layered
            derivatives_t = [comp_P(_derP.params[0], derP.params[0], finP=0, foutderP=0)]
            mP += derivatives_t[0][0][0]  # 1st index = 1st layer, 2nd index select m | d, 3rd index selecting mP | dP
            dP += derivatives_t[0][1][0]
            for _params_layer, params_layer in zip(_derP.params[1:], derP.params[1:]):
                layer_derivatives = comp_ptuple(_params_layer, params_layer)
                derivatives_t += [layer_derivatives]
                mP += layer_derivatives[0][0]
                dP += layer_derivatives[1][0]

    else:  # _derP and derP is params layer
        derivatives_t, dP, mP = comp_ptuple(_derP, derP)


    if foutderP:  # return derP instance
        x0 = min(_derP.x0, derP.x0)
        xn = max(_derP.x0+_derP.L, derP.x0+derP.L)
        L = xn-x0

        dderP = instance(x0=x0, L=L, y=_derP.y, params=derivatives_t, P=derP, _P=_derP)
        return dderP
    else:  # return only the derivatives
        return derivatives_t


def comp_params(_params, params, nparams):

    derivatives, hyps = [[],[]], []

    mP, dP = 0, 0
    for i, (_param, param) in enumerate(zip(_params, params)):
        param_type = i

        if param_type == 0:  # mP | dP
            _mdP = param; mdP = param
            dmdP = _mdP - mdP; mmdP = ave_mP - abs(dmdP)  # need to think on how to get ave_mP or ave_dP, or create another 1?
            derivatives[0].append(mmdP); derivatives[1].append(dmdP)
            dP += dmdP; mP += mmdP

        elif param_type == 1:  # x
            _x = param; x = param
            dx = _x - x; mx = ave_dx - abs(dx)
            derivatives[0].append(dx); derivatives[1].append(mx)
            hyps.append(np.hypot(dx, 1))
            dP += dx; mP += mx

        elif param_type == 2:  # L
            hyp = hyps[i%param_type]
            _L = _param; L = param
            dL = _L - L/hyp;  mL = min(_L, L)
            derivatives[0].append(dL); derivatives[1].append(mL)
            dP += dL; mP += mL

        elif param_type == 3:  # I
            _I = _param; I = param
            dI = _I - I; mI = ave_I - abs(dI)
            derivatives[0].append(dI); derivatives[1].append(mI)
            dP += dI; mP += mI

        elif param_type == 4:  # G
            hyp = hyps[i%param_type]
            _G = _param; G = param
            dG = _G - G/hyp;  mG = min(_G, G)  # if comp_norm: reduce by hypot
            derivatives[0].append(dG); derivatives[1].append(mG)
            dP += dG; mP += mG

        elif param_type == 5:  # Ga
            _Ga = _param; Ga = param
            dGa = _Ga - Ga;  mGa = min(_Ga, Ga)
            derivatives[0].append(dGa); derivatives[1].append(mGa)
            dP += dGa; mP += mGa

        elif param_type == 6:  # M
            hyp = hyps[i%param_type]
            _M = _param; M = param
            dM = _M - M/hyp;  mM = min(_M, M)
            derivatives[0].append(dM); derivatives[1].append(mM)
            dP += dM; mP += mM

        elif param_type == 7:  # Ma
            _Ma = _param; Ma = param
            dMa = _Ma - Ma;  mMa = min(_Ma, Ma)
            derivatives[0].append(dMa); derivatives[1].append(mMa)
            dP += dMa; mP += mMa

        elif param_type == 8:  # angle, (sin_da, cos_da)
            if isinstance(_param, tuple):  # (sin_da, cos_da)
                 _sin_da, _cos_da = _param; sin_da, cos_da = param
                 sin_dda = (cos_da * _sin_da) - (sin_da * _cos_da)  # sin(α - β) = sin α cos β - cos α sin β
                 cos_dda = (cos_da * _cos_da) + (sin_da * _sin_da)  # cos(α - β) = cos α cos β + sin α sin β
                 dangle = (sin_dda, cos_dda)  # da
                 mangle = ave_dangle - abs(np.arctan2(sin_dda, cos_dda))  # ma is indirect match
                 derivatives[0].append(dangle); derivatives[1].append(mangle)
                 dP += np.arctan2(sin_dda, cos_dda); mP += mangle
            else: # m or scalar
                _mangle = _param; mangle = param
                dmangle = _mangle - mangle;  mmangle = min(_mangle, mangle)
                derivatives[0].append(dmangle); derivatives[1].append(mmangle)
                dP += dmangle; mP += mmangle

        elif param_type == 9:  # dangle   (uday, vday, udax, vdax)
            if isinstance(_param, tuple):  # (sin_da, cos_da)
                _uday, _vday, _udax, _vdax = _param
                uday, vday, udax, vdax = param

                sin_dda0 = (vday * _uday) - (uday * _vday)
                cos_dda0 = (vday * _vday) + (uday * _uday)
                sin_dda1 = (vdax * _udax) - (udax * _vdax)
                cos_dda1 = (vdax * _vdax) + (udax * _udax)
                daangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
                # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]
                # dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]
                gay = np.arctan2( (-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
                gax = np.arctan2( (-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?
                maangle = ave_dangle - abs(np.arctan2(gay, gax))  # match between aangles, probably wrong
                derivatives[0].append(daangle); derivatives[1].append(maangle)
                dP += daangle; mP += maangle

            else:  # m or scalar
                _maangle = _param; maangle = param
                dmaangle = _maangle - maangle;  mmaangle = min(_maangle, maangle)
                derivatives[0].append(dmaangle); derivatives[1].append(mmaangle)
                dP += dmaangle; mP += mmaangle

    return derivatives, mP, dP


# draw segments within single dir_blob
def draw_seg_(dir_blob, seg_):
    import random
    import cv2
    import os

    x0 = min([P.x0 for seg in seg_ for P in seg.P__])
    xn = max([P.x0 + P.L for seg in seg_ for P in seg.P__])
    y0 = min([P.y for seg in seg_ for P in seg.P__])
    yn = max([P.y for seg in seg_ for P in seg.P__])

    img = np.zeros((yn - y0 + 1, xn - x0 + 1, 3), dtype="uint8")

    for seg in seg_:
        current_colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        for P in seg.P__:
            img[P.y - y0, P.x0 - x0:P.x0 - x0 + P.L] = current_colour

    cv2.imwrite(os.getcwd() + "/images/comp_slice/img_" + str(dir_blob.id) + ".png", img)

# draw segments within single PP
def draw_PP_segs(dir_blob, PP_):
    import random
    import cv2
    import os

    x0 = min([P.x0 for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])
    xn = max([P.x0 + P.L for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])
    y0 = min([P.y for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])
    yn = max([P.y for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])

    for PP in PP_:
        img = np.zeros((yn - y0 + 1, xn - x0 + 1, 3), dtype="uint8")
        for seg in PP.seg_levels[0]:
            current_colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            for P in seg.P__:
                img[P.y - y0, P.x0 - x0:P.x0 - x0 + P.L] = current_colour

        cv2.imwrite(os.getcwd() + "/images/comp_slice/img_" + str(dir_blob.id) + "_PP_"+str(PP.id)+".png", img)


# draw PPs within single dir_blob
def draw_PPs(dir_blob, PP_, fspliced):
    import random
    import cv2
    import os

    x0 = min([P.x0 for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])
    xn = max([P.x0 + P.L for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])
    y0 = min([P.y for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])
    yn = max([P.y for PP in PP_ for seg in PP.seg_levels[0] for P in seg.P__])

    img = np.zeros((yn - y0 + 1, xn - x0 + 1, 3), dtype="uint8")
    for PP in PP_:
        current_colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for seg in PP.seg_levels[0]:
            for P in seg.P__:
                img[P.y - y0, P.x0 - x0:P.x0 - x0 + P.L] = current_colour

    if fspliced:
        cv2.imwrite(os.getcwd() + "/images/comp_slice/img_" + str(dir_blob.id) + "_PP.png", img)
    else:
        cv2.imwrite(os.getcwd() + "/images/comp_slice/img_" + str(dir_blob.id) + "_PP_spliced.png", img)


def func_layers_chee(_layers, layers, out_layers, func):  # max_nesting = len layers, ntuples = sum(ntuples in lower layers) * 2: 1, 2, 6, 18...

    if _layers and layers:
        if isinstance(_layers[0], list):  # 1st layer is two vertuples, decoded in func; may need recursive unpack if from der+
            sub_out_layers = []
            for _sub_layers, sub_layers in zip(_layers, layers):
                func_layers(_sub_layers, sub_layers, sub_out_layers, func)
            out_layers += [sub_out_layers]
        else:
            func(_layers, layers)  # 1st layer is latuple, decoded in func


def func_layers(_layers, layers, out_layers, func):

    # recursive unpack of nested ptuple pairs, if any from der+, in the bottom layer or sublayer:
    out_layers += [func_pairs(_layers[0], layers[0], out_pairs=[], func_ptuple=func)]

    # recursive unpack of deeper layers, from agg+ in 3rd and higher layers, down to nested tuple pairs
    for _layer, layer in zip(_layers[1:], layers[1:]):
        out_layers += [func_layers(_layer, layer, out_layers, func)]
        # layer = deeper sub_layers
    '''
    1st and 2nd layers are single sublayers, the 2nd adds tuple pair nesting. Both are unpacked by func_pairs, not func_layers.  
    Multiple sublayers start on the 3rd layer, because it's derived from comparison between two (not one) lower layers. 
    4th layer is derived from comparison between 3 lower layers, where the 3rd layer is already nested, etc.
    '''
    return out_layers # possibly nested param layers


def func_pairs(_pairs, pairs, out_pairs, func_ptuple):  # recursively unpack m,d tuple pairs from der+

    if isinstance(_pairs[0], list):  # pairs is a pair, possibly nested
        out_pairs += func_pairs(_pairs[0], pairs[0], out_pairs, func_ptuple)
    else:
        out_pairs += func_ptuple(_pairs[0], pairs[0])  # pairs is actually a ptuple, 1st element is a param

    return out_pairs  # possibly nested m,d ptuple pairs


def sum_layers(summed_params, params):

    sum_pairs(summed_params[0], params[0])  # recursive unpack of nested ptuple pairs, if any from der+, in the bottom layer or sublayer

    for Layer, layer in zip(summed_params[1:], params[1:]):  # recursive unpack of deeper layers, from agg+
        sum_layers(Layer, layer)  # layer = deeper sub_layers


def comp_P(_P, P, instance=CderP, finP=1, foutderP=1):  # forms vertical derivatives of params per P in _P.uplink, conditional ders from norm and DIV comp

    if finP:  # input is CderP
        _P_params = _P.params; P_params = P.params
    else:  # input is param layer
        _P_params = _P; P_params = P

    # compared P params:
    _x, _L, _M, _Ma, _I, _Dx, _Dy, _uday, _vday, _udax, _vdax = _P_params
    x, L, M, Ma, I, Dx, Dy, uday, vday, udax, vdax = P_params

    dx = _x - x;  mx = ave_dx - abs(dx)  # mean x shift, if dx: rx = dx / ((L+_L)/2)? no overlap, offset = abs(x0 -_x0) + abs(xn -_xn)?
    dI = _I - I;  mI = ave_I - abs(dI)
    dM = _M - M;  mM = min(_M, M)
    dMa = _Ma - Ma;  mMa = min(_Ma, Ma)  # dG, dM are directional, re-direct by dx?
    dL = _L - L * np.hypot(dx, 1); mL = min(_L, L)  # if abs(dx) > ave: adjust L as local long axis, no change in G,M
    # G, Ga:
    G = np.hypot(Dy, Dx); _G = np.hypot(_Dy, _Dx)  # compared as scalars
    dG = _G - G;  mG = min(_G, G)
    Ga = (vday + 1) + (vdax + 1); _Ga = (_vday + 1) + (_vdax + 1)  # gradient of angle, +1 for all positives?
    # or Ga = np.hypot( np.arctan2(*Day), np.arctan2(*Dax)?
    dGa = _Ga - Ga;  mGa = min(_Ga, Ga)

    # comp angle:
    _sin = _Dy / (1 if _G==0 else _G); _cos = _Dx / (1 if _G==0 else _G)
    sin  = Dy / (1 if G==0 else G); cos = Dx / (1 if G==0 else G)
    sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
    dangle = np.arctan2(sin_da, cos_da)  # vertical difference between angles
    mangle = ave_dangle - abs(dangle)  # indirect match of angles, not redundant as summed

    # comp angle of angle: forms daa, not gaa?
    sin_dda0 = (vday * _uday) - (uday * _vday)
    cos_dda0 = (vday * _vday) + (uday * _uday)
    sin_dda1 = (vdax * _udax) - (udax * _vdax)
    cos_dda1 = (vdax * _vdax) + (udax * _udax)

    daangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
    # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]
    # dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]
    gay = np.arctan2( (-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
    gax = np.arctan2( (-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?
    daangle = np.arctan2( gay, gax)  # probably wrong
    maangle = ave_daangle - abs(daangle)  # match between aangles, not redundant as summed

    dP = abs(dx)-ave_dx + abs(dI)-ave_I + abs(G)-ave_G + abs(Ga)-ave_Ga + abs(dM)-ave_M + abs(dMa)-ave_Ma + abs(dL)-ave_L
    # sum to evaluate for der+, abs diffs are distinct from directly defined matches:
    mP = mx + mI + mG + mGa + mM + mMa + mL + mangle + maangle

    params = [[mx, mL, mM, mMa, mI, mG, mGa, mangle, maangle, mP],
              [dx, dL, dM, dMa, dI, dG, dGa, dangle, daangle, dP]]

    if foutderP:
        # or summable params only, compute Gs at termination?
        x0 = min(_P.x0, P.x0)
        xn = max(_P.x0+_P.L, P.x0+P.L)
        L = xn-x0
        return instance(x0=x0, L=L, y=_P.y, params=params, P=P, _P=_P)

    else:
        return params
