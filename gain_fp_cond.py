'''
MF solution for 4 pop model with conductance-based synapses
conductance-based visual input for E and P included
'''
import subprocess
import numpy as np

import numpy as np
import time

import sys
sys.path.append('code')
from mf import *

import numpy as np

def pLIFCond0(Re, Rp, Rs, Rv, Rex, be, bp, bs, bv, bex, mu, Cm, gL, Vre, EL, Ee, Ei, Nv, Vlb, Vth):
    # Create mesh values with linear spacing
    vs = np.linspace(Vlb, Vth, Nv)

    # Initialize variables
    p0 = 0
    p0last = 0
    p0int = 0
    v0 = 0

    tauL = Cm / gL
    EPrime = (EL + tauL*Re*(be+be*be)*Ee + tauL*Rex*(bex+bex*bex)*Ee + tauL*Rp*(bp+bp*bp)*Ei + tauL*Rs*(bs+bs*bs)*Ei + tauL*Rv*(bv+bv*bv)*Ei) / (1 + tauL*(Re*(be+be*be) + Rp*(bp+bp*bp) + Rs*(bs+bs*bs) + Rv*(bv+bv*bv)))
    gLPrime = gL * (1 + tauL*(Re*(be+be*be) + Rex*(bex+bex*bex) + Rp*(bp+bp*bp) + Rs*(bs+bs*bs) + Rv*(bv+bv*bv)))

    # Iterate backward through mesh
    for k in range(Nv - 2, -1, -1):
        vleft = vs[k]  # Left endpoint of mesh
        vright = vs[k+1]  # Right endpoint of mesh
        vmid = (vleft + vright) / 2  # Potential at midpoint of mesh
        dv = vright - vleft  # Current mesh step

        # Variance at left, mid, and right points
        Dleft = 0.5 * (Re*be**2 * (vleft - Ee)**2 + Rex*bex**2 * (vleft - Ee)**2 + Rp*bp**2 * (vleft - Ei)**2 + Rs*bs**2 * (vleft - Ei)**2 + Rv*bv**2 * (vleft - Ei)**2)
        Dmid = 0.5 * (Re*be**2 * (vmid - Ee)**2 + Rex*bex**2 * (vmid - Ee)**2 + Rp*bp**2 * (vmid - Ei)**2 + Rs*bs**2 * (vmid - Ei)**2 + Rv*bv**2 * (vmid - Ei)**2)
        Dright = 0.5 * (Re*be**2 * (vright - Ee)**2 + Rex*bex**2 * (vright - Ee)**2 + Rp*bp**2 * (vright - Ei)**2 + Rs*bs**2 * (vright - Ei)**2 + Rv*bv**2 * (vright - Ei)**2)

#         print("Dmid = ",Dmid)
        # Drift and variance terms at left, mid, and right points
        Gleft = -(-gLPrime*(vleft - EPrime) + mu) / (Dleft * Cm)
        Gmid = -(-gLPrime*(vmid - EPrime) + mu) / (Dmid * Cm)
        Gright = -(-gLPrime*(vright - EPrime) + mu) / (Dright * Cm)

        # Simpson's rule integration for Gint2 (from left to midpoint) and Gint (over the full interval)
        vmidmid = (vleft + vmid) / 2
        Dmidmid = 0.5 * (Re*be**2 * (vmidmid - Ee)**2 + Rex*bex**2 * (vmidmid - Ee)**2 + Rp*bp**2 * (vmidmid - Ei)**2 + Rs*bs**2 * (vmidmid - Ei)**2 + Rv*bv**2 * (vmidmid - Ei)**2)
        Gmidmid = -(-gLPrime*(vmidmid - EPrime) + mu) / (Dmidmid * Cm)

        Gint2 = (1.0 / 6.0) * (Gleft + 4 * Gmidmid + Gmid) * (vmid - vleft)
        Gint = (1.0 / 6.0) * (Gleft + 4 * Gmid + Gright) * dv

        # Simpson's rule to integrate H * exp(Int G)
        H = (vmid > Vre) * (1 / Dmid)
        HexpIntG = (1.0 / 6.0) * H * (1 + 4 * np.exp(Gint2) + np.exp(Gint)) * dv

        # Update p0
        p0 = p0 * np.exp(Gint) + HexpIntG

        # Trapezoidal approximation to integral
        p0int += dv * ((p0 + p0last) / 2.0)
        v0 += dv * ((p0 * vleft + p0last * vright) / 2.0)
        p0last = p0

    r0 = 1 / p0int  # rate is 1/(integral of p0)
    v0 = v0 * r0  # membrane potential is (integral of p0*v)*rate

    return r0, v0

def pfind_fp_4pop(nu, j, vis, max_step=100, N=1e6, qq=0.1, tolerance = 1e-3):
    nu_e, nu_p, nu_s, nu_v = nu
    j_e, j_p, j_s, j_v = j
    vis_e, vis_p = vis
    step = 0

    mu_e_0 = np.zeros(xpoints)
    mu_p_0 = np.zeros(xpoints)
    mu_s_0 = np.zeros(xpoints)
    mu_v_0 = np.zeros(xpoints)

    D_e_0 = np.zeros(xpoints)
    D_p_0 = np.zeros(xpoints)
    D_s_0 = np.zeros(xpoints)
    D_v_0 = np.zeros(xpoints)

    sqrtN = np.sqrt(N)


    Nv = 1000
    Vlb = -1.
    Vth = 1.
    Ee = 1.5
    Ei = -0.5
    
    bee = 1. - np.exp(-j_ee/sqrtN)
    bpe = 1. - np.exp(-j_pe/sqrtN)
    bse = 1. - np.exp(-j_se/sqrtN)
    bve = 1. - np.exp(-j_ve/sqrtN)
    
    bep = 1. - np.exp(-j_ep/sqrtN)
    bpp = 1. - np.exp(-j_pp/sqrtN)
    bsp = 1. - np.exp(-j_sp/sqrtN)
    bvp = 1. - np.exp(-j_vp/sqrtN)
    
    bes = 1. - np.exp(-j_es/sqrtN)
    bps = 1. - np.exp(-j_ps/sqrtN)
    bss = 1. - np.exp(-j_ss/sqrtN)
    bvs = 1. - np.exp(-j_vs/sqrtN)
    
    bev = 1. - np.exp(-j_ev/sqrtN)
    bpv = 1. - np.exp(-j_pv/sqrtN)
    bsv = 1. - np.exp(-j_sv/sqrtN)
    bvv = 1. - np.exp(-j_vv/sqrtN)
    
    print(bee,bpe,bse,bve)
    
    status = 1

    bex =  1. - np.exp(-0.2)
    bpx =  1. - np.exp(-0.02)

    while step < max_step:

        step += 1
        print('step ',step, '... ', end='')

        # compute mean inputs
        Re_rolled = []
        Rp_rolled = []
        Rs_rolled = []
        Rv_rolled = []
        
        for j in range(xpoints):
            Re_j = N*q_e*np.mean(nu_e*np.roll(proj_e, j))
            Re_rolled.append(Re_j)
            Rp_j = N*q_p*np.mean(nu_p*np.roll(proj_p, j))
            Rp_rolled.append(Rp_j)
            Rs_j = N*q_s*np.mean(nu_s*np.roll(proj_s, j))
            Rs_rolled.append(Rs_j)
            Rv_j = N*q_v*np.mean(nu_v*np.roll(proj_v, j))
            Rv_rolled.append(Rv_j)
            
        arrs = [mu_e_0, mu_p_0, mu_s_0, mu_v_0, D_e_0, D_p_0, D_s_0, D_v_0]
        names = ['mu_e', 'mu_p', 'mu_s', 'mu_v', 'D_e', 'D_p', 'D_s', 'D_v']
        for i in range(len(names)):
            if np.any(np.isnan(arrs[i])):
                print(names[i], 'is nan')

        # compute next iteration 
            
        nu1_e = np.zeros(xpoints)
        nu1_p = np.zeros(xpoints)
        nu1_s = np.zeros(xpoints)
        nu1_v = np.zeros(xpoints)
        
        for j in range(xpoints):

            
            nu1_e[j], _ = pLIFCond0(kbar_ee*Re_rolled[j], kbar_ep*Rp_rolled[j], kbar_es*Rs_rolled[j], kbar_ev*Rv_rolled[j], vis_e[j],
                                bee, bep, bes, bev, bex, sqrtN*j_e[j], Cme, gLe, Vre, EL, Ee, Ei, Nv, Vlb, Vth)

            nu1_p[j], _ = pLIFCond0(kbar_pe*Re_rolled[j], kbar_pp*Rp_rolled[j], kbar_ps*Rs_rolled[j], kbar_pv*Rv_rolled[j], vis_p[j],
                                bpe, bpp, bps, bpv, bpx, sqrtN*j_p[j], Cmp, gLp, Vre, EL, Ee, Ei, Nv, Vlb, Vth)

            nu1_s[j], _ = pLIFCond0(kbar_se*Re_rolled[j], kbar_sp*Rp_rolled[j], kbar_ss*Rs_rolled[j], kbar_sv*Rv_rolled[j], 0,
                                bse, bsp, bss, bsv, 0, sqrtN*j_s[j], Cms, gLs, Vre, EL, Ee, Ei, Nv, Vlb, Vth)
            
            nu1_v[j], _ = pLIFCond0(kbar_ve*Re_rolled[j], kbar_vp*Rp_rolled[j], kbar_vs*Rs_rolled[j], kbar_vv*Rv_rolled[j], 0,
                                bve, bvp, bvs, bvv, 0, sqrtN*j_v[j], Cmv, gLv, Vre, EL, Ee, Ei, Nv, Vlb, Vth)

            
        nancase = False
        if np.any(np.isnan(nu1_e)):
            print('WARNING: nan in nu1_e')
            print(nu1_e)
            nu1_e[np.isnan(nu1_e)] = 0
            nancase = True
        if np.any(np.isnan(nu1_p)):
            print('WARNING: nan in nu1_p')
            print(nu1_p)
            nu1_p[np.isnan(nu1_p)] = 0
            nancase = True
        if np.any(np.isnan(nu1_s)):
            print('WARNING: nan in nu1_s')
            print(nu1_s)
            nu1_s[np.isnan(nu1_s)] = 0
            nancase = True
        if np.any(np.isnan(nu1_v)):
            print('WARNING: nan in nu1_v')
            print(nu1_v)
            nu1_v[np.isnan(nu1_v)] = 0
            nancase = True
            
            
        dev_e = l2_deviation(nu1_e, nu_e)
        dev_p = l2_deviation(nu1_p, nu_p)
        dev_s = l2_deviation(nu1_s, nu_s)
        dev_v = l2_deviation(nu1_v, nu_v)
#         if dev_e+dev_p+dev_s < 3*tolerance and not nancase:
        if dev_e+dev_p+dev_s+dev_v < 4*tolerance:
            print('Converged after',step,'steps.')
            nu_e = nu1_e
            nu_p = nu1_p
            nu_s = nu1_s
            nu_v = nu1_v
            status = 0
            break

        else:
            print(dev_e, dev_p, dev_s, dev_v)
            nu_e = nu1_e*qq + nu_e*(1.-qq)
            nu_p = nu1_p*qq + nu_p*(1.-qq)
            nu_s = nu1_s*qq + nu_s*(1.-qq)
            nu_v = nu1_v*qq + nu_v*(1.-qq)
            continue

    return nu_e, nu_p, nu_s, nu_v, arrs, status



parname = sys.argv[1]
param = np.load('mf_results/%s/pars.npy'%parname)
stim_type = sys.argv[2]


#unpack
taume, taump, taums, taumv, sigma_e, sigma_p, sigma_s, sigma_v,\
           j_ee, j_ep, j_es, j_ev, j_pe, j_pp, j_ps, j_pv, j_se, j_sp, j_ss, j_sv, j_ve, j_vp, j_vs, j_vv,\
           q_e, q_p, q_s, q_v, epsilon, \
           kbar_ee, kbar_ep, kbar_es,kbar_ev,\
           kbar_pe, kbar_pp, kbar_ps, kbar_pv,\
           kbar_se, kbar_sp, kbar_ss, kbar_sv,\
           kbar_ve, kbar_vp, kbar_vs, kbar_vv,\
           sigma_o, p_e, p_p, p_s, p_v, jbar_e, jbar_p, jbar_s, jbar_v = param

# constants
Vth = 1
Vre = 0
Vlb = -1
EL = 0.

Cme = 1
gLe = Cme / taume
Cmp = 1
gLp = Cmp / taump
Cms = 1
gLs = Cms / taums
Cmv = 1
gLv = Cmv / taumv
print(taume,taump,taums, taumv)

dv0 = 0.01
vs = np.arange(Vlb, Vth, dv0)

# spatial resolution
x_res = 0.005
xpoints = int(1/x_res)

# projections
x, proj_e = wrapped_gaussian(0., sigma_e, npoints=2, xpoints=xpoints, xleft=0, xright=1)
x, proj_p = wrapped_gaussian(0., sigma_p, npoints=2, xpoints=xpoints, xleft=0, xright=1)
x, proj_s = wrapped_gaussian(0., sigma_s, npoints=2, xpoints=xpoints, xleft=0, xright=1)
x, proj_v = wrapped_gaussian(0., sigma_v, npoints=2, xpoints=xpoints, xleft=0, xright=1)

# synaptic weights
j_mat = np.array([[j_ee, j_ep, j_es, j_ev], [j_pe, j_pp, j_ps, j_pv], 
                  [j_se, j_sp, j_ss, j_sv], [j_ve, j_vp, j_vs, j_vv]])
q_mat = np.array([[q_e, q_p, q_s, q_v] for _ in range(4)])


# connection probabilities
kbar_mat = np.array([[kbar_ee, kbar_ep, kbar_es, kbar_ev], [ kbar_pe, kbar_pp, kbar_ps, kbar_pv],
                     [kbar_se, kbar_sp, kbar_ss, kbar_sv], [kbar_ve, kbar_vp, kbar_vs, kbar_vv]])

wbar_mat = q_mat * j_mat * kbar_mat
wbar_ee = wbar_mat[0,0]
wbar_ep = wbar_mat[0,1]
wbar_es = wbar_mat[0,2]
wbar_ev = wbar_mat[0,3]
wbar_pe = wbar_mat[1,0]
wbar_pp = wbar_mat[1,1]
wbar_ps = wbar_mat[1,2]
wbar_pv = wbar_mat[1,3]
wbar_se = wbar_mat[2,0]
wbar_sp = wbar_mat[2,1]
wbar_ss = wbar_mat[2,2]
wbar_sv = wbar_mat[2,3]
wbar_ve = wbar_mat[3,0]
wbar_vp = wbar_mat[3,1]
wbar_vs = wbar_mat[3,2]
wbar_vv = wbar_mat[3,3]

#external inputs
_, j_e = wrapped_gaussian(0.5, sigma_o, npoints=2, xpoints=xpoints, xleft=0, xright=1)
_, j_p = wrapped_gaussian(0.5, sigma_o, npoints=2, xpoints=xpoints, xleft=0, xright=1)
_, j_s = wrapped_gaussian(0.5, sigma_o, npoints=2, xpoints=xpoints, xleft=0, xright=1)
_, j_v = wrapped_gaussian(0.5, sigma_o, npoints=2, xpoints=xpoints, xleft=0, xright=1)

j_e = p_e*jbar_e*j_e + (1-p_e)*jbar_e
j_p = p_p*jbar_p*j_p + (1-p_p)*jbar_p
j_s = p_s*jbar_s*j_s + (1-p_s)*jbar_s
j_v = p_v*jbar_v*j_v + (1-p_v)*jbar_v


import pickle
N=300000

#for param12
N=100000

with open(f'mf_results/{parname}/N_{int(N)}.pkl','rb') as f:
    x, nu_e, nu_p, nu_s, nu_v = pickle.load(f)


# flat spontaneous activity
p_e = 0.
p_p = 0.
p_s = 0.
p_v = 0.

pv_stim, sst_stim, vip_stim = 0.0, 0.0, 0.0
if stim_type == 'sst_distal':
    chr2_strength = float(sys.argv[3])
    sst_stim = chr2_strength*jbar_s
    stim_location = 0.1
    output_file_name = f'mf_results/{parname}/N_{int(N)}_{stim_type}_{chr2_strength}.pkl'
elif stim_type == 'pv_distal':
    chr2_strength = float(sys.argv[3])
    pv_stim = chr2_strength*jbar_p
    stim_location = 0.1
    output_file_name = f'mf_results/{parname}/N_{int(N)}_{stim_type}_{chr2_strength}.pkl'
elif stim_type == 'vip_distal':
    chr2_strength = float(sys.argv[3])
    sst_stim = 0*jbar_s
    vip_stim = chr2_strength*jbar_v
    stim_location = 0.1
    output_file_name = f'mf_results/{parname}/N_{int(N)}_{stim_type}_{chr2_strength}.pkl'
elif stim_type == 'sst_proximal':
    chr2_strength = float(sys.argv[3])
    sst_stim = chr2_strength*jbar_s
    stim_location = 0.5
    output_file_name = f'mf_results/{parname}/N_{int(N)}_{stim_type}_{chr2_strength}.pkl'
elif stim_type == 'pv_proximal':
    chr2_strength = float(sys.argv[3])
    pv_stim = chr2_strength*jbar_p
    stim_location = 0.5
    output_file_name = f'mf_results/{parname}/N_{int(N)}_{stim_type}_{chr2_strength}.pkl'
elif stim_type == 'vip_proximal':
    chr2_strength = float(sys.argv[3])
    vip_stim = chr2_strength*jbar_v
    stim_location = 0.5
    output_file_name = f'mf_results/{parname}/N_{int(N)}_{stim_type}_{chr2_strength}.pkl'
elif stim_type == 'sst_pos':
    chr2_strength = float(sys.argv[3])
    stim_location = float(sys.argv[4])
    sst_stim = chr2_strength*jbar_s
    output_file_name = f'mf_results/{parname}/N_{int(N)}_{stim_type}_{stim_location:.1f}_{chr2_strength:.2f}.pkl'
elif stim_type == 'pv_pos':
    chr2_strength = float(sys.argv[3])
    stim_location = float(sys.argv[4])
    pv_stim = chr2_strength*jbar_p
    output_file_name = f'mf_results/{parname}/N_{int(N)}_{stim_type}_{stim_location:.1f}_{chr2_strength:.2f}.pkl'
elif stim_type == 'vip_pos':
    chr2_strength = float(sys.argv[3])
    stim_location = float(sys.argv[4])
    vip_stim = chr2_strength*jbar_v
    output_file_name = f'mf_results/{parname}/N_{int(N)}_{stim_type}_{stim_location:.1f}_{chr2_strength:.2f}.pkl'
elif stim_type == 'control':
    stim_location = .0
    output_file_name = f'mf_results/{parname}/N_{int(N)}_{stim_type}.pkl'
else:
    raise ValueError('stim_type not recognized')

sigma_o = 0.2

nus = []
visual_stims = [0.01*i for i in range(31)]
for visual_stim in visual_stims:
    print('running: ', parname, stim_type)
    print('visual_stim = ',visual_stim)
    
    
    # constants
    Vth = 1
    Vre = 0
    Vlb = -1
    EL = 0.

    Cme = 1
    gLe = Cme / taume
    Cmp = 1
    gLp = Cmp / taump
    Cms = 1
    gLs = Cms / taums
    Cmv = 1
    gLv = Cmv / taumv
    print(taume,taump,taums, taumv)

    dv0 = 0.01
    vs = np.arange(Vlb, Vth, dv0)

    # spatial resolution
    x_res = 0.005
    xpoints = int(1/x_res)

    # projections
    x, proj_e = wrapped_gaussian(0., sigma_e, npoints=2, xpoints=xpoints, xleft=0, xright=1)
    x, proj_p = wrapped_gaussian(0., sigma_p, npoints=2, xpoints=xpoints, xleft=0, xright=1)
    x, proj_s = wrapped_gaussian(0., sigma_s, npoints=2, xpoints=xpoints, xleft=0, xright=1)
    x, proj_v = wrapped_gaussian(0., sigma_v, npoints=2, xpoints=xpoints, xleft=0, xright=1)

    # synaptic weights
    j_mat = np.array([[j_ee, j_ep, j_es, j_ev], [j_pe, j_pp, j_ps, j_pv], 
                      [j_se, j_sp, j_ss, j_sv], [j_ve, j_vp, j_vs, j_vv]])
    q_mat = np.array([[q_e, q_p, q_s, q_v] for _ in range(4)])


    # connection probabilities
    kbar_mat = np.array([[kbar_ee, kbar_ep, kbar_es, kbar_ev], [ kbar_pe, kbar_pp, kbar_ps, kbar_pv],
                         [kbar_se, kbar_sp, kbar_ss, kbar_sv], [kbar_ve, kbar_vp, kbar_vs, kbar_vv]])

    wbar_mat = q_mat * j_mat * kbar_mat
    wbar_ee = wbar_mat[0,0]
    wbar_ep = wbar_mat[0,1]
    wbar_es = wbar_mat[0,2]
    wbar_ev = wbar_mat[0,3]
    wbar_pe = wbar_mat[1,0]
    wbar_pp = wbar_mat[1,1]
    wbar_ps = wbar_mat[1,2]
    wbar_pv = wbar_mat[1,3]
    wbar_se = wbar_mat[2,0]
    wbar_sp = wbar_mat[2,1]
    wbar_ss = wbar_mat[2,2]
    wbar_sv = wbar_mat[2,3]
    wbar_ve = wbar_mat[3,0]
    wbar_vp = wbar_mat[3,1]
    wbar_vs = wbar_mat[3,2]
    wbar_vv = wbar_mat[3,3]

    #external inputs
    _, visual_input = wrapped_gaussian(0.5, 0.2, npoints=2, xpoints=xpoints, xleft=0, xright=1)
    
    _, chr2 = wrapped_gaussian(stim_location, 0.2, npoints=2, xpoints=xpoints, xleft=0, xright=1)

    j_e = np.array([jbar_e for _ in range(xpoints)]) 
    j_p = np.array([jbar_p for _ in range(xpoints)]) + pv_stim*chr2
    j_s = np.array([jbar_s for _ in range(xpoints)]) + sst_stim*chr2
    j_v = np.array([jbar_v for _ in range(xpoints)]) + vip_stim*chr2

    vis_e = visual_stim * (40./1000.) * visual_input
    vis_p = visual_stim * (10./1000.) * visual_input
    
    nu_e, nu_p, nu_s, nu_v, arrs, status = pfind_fp_4pop([nu_e, nu_p, nu_s, nu_v], [j_e, j_p, j_s, j_v], [vis_e, vis_p], max_step=1000, N=N, qq=0.05, tolerance = 1e-5)
    if status == 0:
        nus.append([nu_e.copy(), nu_p.copy(), nu_s.copy(), nu_v.copy()])
        with open(output_file_name,'wb') as f:
            pickle.dump([visual_stims, nus], f)
    else:
        print('no convergence')
        break
        
with open(output_file_name,'wb') as f:
    pickle.dump([visual_stims, nus], f)
