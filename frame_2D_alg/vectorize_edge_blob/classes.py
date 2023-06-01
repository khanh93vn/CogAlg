from class_cluster import ClusterStructure, init_param

class CQ(ClusterStructure):  # generic links

    Q : list = init_param([])  # generic sequence or index increments in ptuple, derH, etc
    Qm : list = init_param([])  # in-graph only
    Qd : list = init_param([])
    ext : list = init_param([[], []])  # [ms,ds], per subH only
    valt : list = init_param([0,0])  # in-graph vals
    rdnt : list = init_param([1,1])  # none if represented m and d?
    out_valt : list = init_param([0,0])  # of non-graph links, as alt?
    fds : list = init_param([])  # not used?
    rng : int = 1  # not used?

class CH(ClusterStructure):  # generic hierarchy, or that's most of a node?
      pass

class Cptuple(ClusterStructure):  # bottom-layer tuple of compared params in P, derH per par in derP, or PP

    I : int = 0  # [m,d] in higher layers:
    M : int = 0
    Ma : float = 0.0
    angle : list = init_param([0,0])  # in latuple only, replaced by float in vertuple
    aangle : list = init_param([0,0,0,0])
    G : float = 0.0  # for comparison, not summation:
    Ga : float = 0.0
    L : int = 0  # replaces n, still redundant to len dert_ in P, nlinks in PP or graph


class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple : list = init_param([])  # latuple: I, M, Ma, G, Ga, angle(Dy, Dx), aangle( Sin_da0, Cos_da0, Sin_da1, Cos_da1)
    derT : list = init_param([[],[]])  # ptuple) fork) layer) H)T:  1ptuple, 1fork, 1layer in comp_slice, extend in der+ and fb
    valT : list = init_param([0,0])
    rdnT : list = init_param([1,1])
    axis : list = init_param([0,0])  # prior slice angle, init sin=0,cos=1
    box : list = init_param([0,0,0,0])  # y0,yn, x0,xn
    dert_ : list = init_param([])  # array of pixel-level derts, redundant to uplink_, only per blob?
    link_ : list = init_param([])  # all links
    link_t : list = init_param([[],[]])  # +ve rlink_, dlink_
    roott : list = init_param([None, None])  # m,d PP that contain this P
    dxdert_ : list = init_param([])  # only in Pd
    Pd_ : list = init_param([])  # only in Pm
    # if comp_dx:
    Mdx : int = 0
    Ddx : int = 0

class CderP(ClusterStructure):  # tuple of derivatives in P link: binary tree with latuple root and vertuple forks

    derT : list = init_param([])  # vertuple_ per layer, unless implicit? sum links / rng+, layers / der+?
    valT : list = init_param([0,0])  # also of derH
    rdnT : list = init_param([1,1])  # mrdn + uprdn if branch overlap?
    _P : object = None  # higher comparand
    P : object = None  # lower comparand
    roott : list = init_param([None, None])  # for der++
    box : list = init_param([0,0,0,0])  # y0,yn, x0,xn: P.box+_P.box, or center+_center?
    L : int = 0
    fdx : object = None  # if comp_dx
'''
max ntuples / der layer = ntuples in lower layers: 1, 1, 2, 4, 8...
lay1: par     # derH per param in vertuple, each layer is derivatives of all lower layers:
lay2: [m,d]   # implicit nesting, brackets for clarity:
lay3: [[m,d], [md,dd]]: 2 sLays,
lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays:
'''

class CPP(CderP):

    ptuple : list = init_param([])  # summed P__ ptuples, = 0th derLay
    derT : list = init_param([[],[]])  # ptuple) fork) layer) H)T: 1ptuple, 1fork, 1layer in comp_slice, extend in sub+ and fb
    valT : list = init_param([[],[]])  # per derT( H( layer( fork
    rdnT : list = init_param([[],[]])
    fd : int = 0  # global?
    rng : int = 1
    box : list = init_param([0,0,0,0])  # y0,yn, x0,xn
    mask__ : object = None
    P__ : list = init_param([])  # 2D array of nodes: Ps or sub-PPs
    link_ : list = init_param([])  # all links summed from Ps
    link_t : list = init_param([[],[]])  # +ve rlink_, dlink_
    roott : list = init_param([None, None])  # PPPm|PPPd containing this PP
    cPP_ : list = init_param([])  # rdn reps in other PPPs, to eval and remove?
    fb_ : list = init_param([])  # [[new_ders,val,rdn]]: [feedback per node]
    Rdn : int = 0  # for accumulation or separate recursion count?
    # fdiv = NoneType  # if div_comp?

class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers
    ''' ext / agg.sub.derH:
    L : list = init_param([])  # der L, init None
    S : int = 0  # sparsity: ave len link
    A : list = init_param([])  # area|axis: Dy,Dx, ini None
    '''
    G : object = None  # same-scope lower-der|rng G.G.G., or [G0,G1] in derG, None in PP
    root : object = None  # root graph or derH G, element of ex.H[-1][fd]
    pH : list = init_param([])  # aggH( subH( derH H: Lev+= node tree slice/fb, Lev/agg+, lev/sub+?  subH if derG
    H : list = init_param([])  # replace with node_ per pH[i]? down-forking tree of Levs: slice of nodes
    # uH: up-forking Levs if mult roots
    node_ : list = init_param([])  # single-fork, conceptually H[0], concat sub-node_s in ex.H levs
    link_ : CQ = init_param(CQ())  # temporary holder for der+ node_, then unique links within graph?
    valT : list = init_param([0,0])
    rdnt : list = init_param([1,1])
    fterm : int = 0  # node_ sub-comp was terminated
    rng : int = 1
    box : list = init_param([0,0,0,0,0,0])  # y,x, y0,yn, x0,xn
    nval : int = 0  # of open links: base alt rep
    alt_graph_ : list = init_param([])  # contour + overlapping contrast graphs
    alt_Graph : object = None  # conditional, summed and concatenated params of alt_graph_