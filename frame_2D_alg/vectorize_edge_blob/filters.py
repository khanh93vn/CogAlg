ave_inv = 20  # ave inverse m, change to Ave from the root intra_blob?
ave = 5  # ave direct m, change to Ave_min from the root intra_blob?
ave_g = 30  # change to Ave from the root intra_blob?
ave_ga = 0.78  # ga at 22.5 degree
flip_ave = .1
flip_ave_FPP = 0  # flip large FPPs only (change to 0 for debug purpose)
div_ave = 200
ave_rmP = .7  # the rate of mP decay per relative dX (x shift) = 1: initial form of distance
ave_ortho = 20
aveB = 50  # same for m and d?
ave_x = 1
ave_dx = 5  # inv, difference between median x coords of consecutive Ps
ave_dy = 5
ave_daxis = .2
ave_dangle = .2  # vertical difference between angles: -1->1, abs dangle: 0->1, ave_dangle = (min abs(dangle) + max abs(dangle))/2,
# max mangle = ave_dangle + 1; or ave_dangle is from comp adjacent angles, which is lower?
ave_daangle = 2

ave_mval = ave_dval = 10  # should be different
vaves = [ave_mval, ave_dval]
ave_Pm = ave_Pd = 10
P_aves = [ave_Pm,ave_Pd]
ave_PPm = ave_PPd = 30
PP_aves = [ave_PPm,ave_PPd]
ave_Gm = ave_Gd = 50
G_aves = [ave_Gm,ave_Gd]

ave_splice = 10
ave_nsubt = [3,3]
ave_sub = 2  # cost of calling sub_recursion and looping
ave_agg = 3  # cost of agg_recursion
ave_overlap = 10
ave_dist = 3
max_dist = 15
med_decay = .5  # decay of induction per med layer
PP_vars = ["I", "M", "Ma", "angle", "aangle", "G", "Ga", "x", "L"]
# comp_param coefs:
ave_dI = ave_inv
ave_mI = ave # replace the rest with coefs:
ave_mG = 10
ave_mM = 2
ave_mMa = .1
ave_mA = .2
ave_mL = 2
aves = [ave_mI, ave_mG, ave_mM, ave_mMa, ave_mA, ave_mL]
ave_rotate = 10
