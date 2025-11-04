import subprocess
import numpy as np
import pickle
import os
import time

import sys

parname = sys.argv[1]
N = int(float(sys.argv[2]))

def wrapped_gaussian(mu,sigma,npoints=2,xpoints=100,xleft=0,xright=1):
    x=np.linspace(xleft,xright,xpoints+1)[:-1]
#     mu = 0.5
#     sigma = 0.5
#     npoints = 2
    xar = np.array([x+n for n in range(-npoints,npoints+1)])
    xn = np.exp(-np.power(xar-mu,2)/(2*sigma*sigma))
    #     print(xn)
    gx = np.sum(xn, axis=0)/np.sqrt(2*np.pi)/sigma
    return x, gx

def l2_deviation(vector1, vector2):
    difference = np.array(vector1) - np.array(vector2)
    l2_dev = np.sqrt(np.sum(difference ** 2))
    return l2_dev

def EIF0(input_parameters):

    input_string = ' '.join(map(str, input_parameters))

    try:
        result = subprocess.check_output(["./code/pEIF0.out"], input=input_string, text=True, stderr=subprocess.PIPE)
        return float(result.split('\n')[0].split()[1])
    except subprocess.CalledProcessError as e:
        print(f"Error running C program: {e}")
        print("Error Output:")
        print(e.stderr)
        
def LIF0(input_parameters, printout=False):

    input_string = ' '.join(map(str, input_parameters))

    try:
        result = subprocess.check_output(["./code/pLIF0.out"], input=input_string, text=True, stderr=subprocess.PIPE)
        if printout:
            print(result)
        return float(result.split('\n')[0].split()[1])
    except subprocess.CalledProcessError as e:
        print(f"Error running C program: {e}")
        print("Error Output:")
        print(e.stderr)

def find_fp_4pop(nu, j, max_step=100, N=1e6, qq=0.1, tolerance = 1e-3):
    nu_e, nu_p, nu_s, nu_v = nu
    j_e, j_p, j_s, j_v = j
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
    
    print('qp=%.2f;'%q_p)
    print('qs=%.2f;'%q_s)
    print('sigmaee=%.2f;'%sigma_e)
    print('sigmaep=%.2f;'%sigma_p)
    print('sigmaes=%.2f;'%sigma_s)
    print('sigmape=%.2f;'%sigma_e)
    print('sigmapp=%.2f;'%sigma_p)
    print('sigmaps=%.2f;'%sigma_s)
    print('sigmase=%.2f;'%sigma_e)
    print('sigmasp=%.2f;'%sigma_p)
    print('sigmass=%.2f;'%sigma_s)

    print('sigmaExte=%.2f;'%sigma_o)
    print('sigmaExtp=%.2f;'%sigma_o)
    print('sigmaExts=%.2f;'%sigma_o)
    print('N=%s'%N)

    print('kbar_ee=%.3f'%kbar_ee)
    print('kbar_ep=%.3f'%kbar_ep)
    print('kbar_es=%.3f'%kbar_es)
    print('kbar_pe=%.3f'%kbar_pe)
    print('kbar_pp=%.3f'%kbar_pp)
    print('kbar_ps=%.3f'%kbar_ps)
    print('kbar_se=%.3f'%kbar_se)
    print('kbar_sp=%.3f'%kbar_sp)
    print('kbar_ss=%.3f'%kbar_ss)
    print('knorm=1.')

    print('jee=%s'%j_ee)
    print('jep=%s'%j_ep)
    print('jes=%s'%j_es)
    print('jpe=%s'%j_pe)
    print('jpp=%s'%j_pp)
    print('jps=%s'%j_ps)
    print('jse=%s'%j_se)
    print('jsp=%s'%j_sp)
    print('jss=%s'%j_ss)

    print('jExte=%.5f'%jbar_e)
    print('jExtp=%.5f'%jbar_p)
    print('jExts=%.5f'%jbar_s)

    print('pExte=%.3f'%p_e)
    print('pExtp=%.3f'%p_p)
    print('pExts=%.3f'%p_s)

    print('Vth=%.2f;'%(Vth))
    print('Vre=%.2f;'%Vre)
    print('Vlb=%.2f;'%Vlb)
    print('taume=%.2f;'%taume)
    print('taump=%.2f;'%taump)
    print('taums=%.2f;'%taums)


    
    status = 1

    while step < max_step:

        step += 1
        print('step ',step, '... ', end='')

        # compute mean inputs
        nu_e_rolled = []
        nu_p_rolled = []
        nu_s_rolled = []
        nu_v_rolled = []
        mu_e_0 = np.zeros(xpoints)
        for j in range(xpoints):
            nu_e_rolled_j = nu_e*np.roll(proj_e, j)
            nu_e_rolled.append(nu_e_rolled_j)
            nu_p_rolled_j = nu_p*np.roll(proj_p, j)
            nu_p_rolled.append(nu_p_rolled_j)
            nu_s_rolled_j = nu_s*np.roll(proj_s, j)
            nu_s_rolled.append(nu_s_rolled_j)
            nu_v_rolled_j = nu_v*np.roll(proj_v, j)
            nu_v_rolled.append(nu_v_rolled_j)
            mu_e_0[j] = sqrtN*(wbar_ee*np.mean(nu_e_rolled_j) - wbar_ep*np.mean(nu_p_rolled_j) - wbar_es*np.mean(nu_s_rolled_j) - wbar_ev*np.mean(nu_v_rolled_j) + j_e[j])

        mu_p_0 = np.zeros(xpoints)
        for j in range(xpoints):
            mu_p_0[j] = sqrtN*(wbar_pe*np.mean(nu_e_rolled[j]) - wbar_pp*np.mean(nu_p_rolled[j]) - wbar_ps*np.mean(nu_s_rolled[j]) - wbar_pv*np.mean(nu_v_rolled[j]) + j_p[j])

        mu_s_0 = np.zeros(xpoints)
        for j in range(xpoints):
            mu_s_0[j] = sqrtN*(wbar_se*np.mean(nu_e_rolled[j]) - wbar_sp*np.mean(nu_p_rolled[j]) - wbar_ss*np.mean(nu_s_rolled[j]) - wbar_sv*np.mean(nu_v_rolled[j]) + j_s[j])
            
        mu_v_0 = np.zeros(xpoints)
        for j in range(xpoints):
            mu_v_0[j] = sqrtN*(wbar_ve*np.mean(nu_e_rolled[j]) - wbar_vp*np.mean(nu_p_rolled[j]) - wbar_vs*np.mean(nu_s_rolled[j]) - wbar_vv*np.mean(nu_v_rolled[j]) + j_v[j])


        # compute Diffusion 
        D_e_0 = np.zeros(xpoints)
        for j in range(xpoints):
            D_e_0[j] = (j_ee*wbar_ee*np.mean(nu_e_rolled[j]) + j_ep*wbar_ep*np.mean(nu_p_rolled[j]) + j_es*wbar_es*np.mean(nu_s_rolled[j]) + j_ev*wbar_ev*np.mean(nu_v_rolled[j]))/2.

        D_p_0 = np.zeros(xpoints)
        for j in range(xpoints):
            D_p_0[j] = (j_pe*wbar_pe*np.mean(nu_e_rolled[j]) + j_pp*wbar_pp*np.mean(nu_p_rolled[j]) + j_ps*wbar_ps*np.mean(nu_s_rolled[j]) + j_pv*wbar_pv*np.mean(nu_v_rolled[j]))/2.

        D_s_0 = np.zeros(xpoints)
        for j in range(xpoints):
            D_s_0[j] = (j_se*wbar_se*np.mean(nu_e_rolled[j]) + j_sp*wbar_sp*np.mean(nu_p_rolled[j]) + j_ss*wbar_ss*np.mean(nu_s_rolled[j]) + j_sv*wbar_sv*np.mean(nu_v_rolled[j]))/2.
   
        D_v_0 = np.zeros(xpoints)
        for j in range(xpoints):
            D_v_0[j] = (j_ve*wbar_ve*np.mean(nu_e_rolled[j]) + j_vp*wbar_vp*np.mean(nu_p_rolled[j]) + j_vs*wbar_vs*np.mean(nu_s_rolled[j]) + j_vv*wbar_vv*np.mean(nu_v_rolled[j]))/2.
 
        arrs = [mu_e_0, mu_p_0, mu_s_0, mu_v_0, D_e_0, D_p_0, D_s_0, D_v_0]
        names = ['mu_e', 'mu_p', 'mu_s', 'mu_v', 'D_e', 'D_p', 'D_s', 'D_v']
        for i in range(len(names)):
            if np.any(np.isnan(arrs[i])):
                print(names[i], 'is nan')

        # compute next iteration 
            
        nu1_e = np.zeros(xpoints)
        for j in range(xpoints):
            nu1_e_j = LIF0([mu_e_0[j], D_e_0[j], Cme, gLe, Vre, EL, Nv, Vlb, Vth])
            nu1_e[j] = nu1_e_j

        nu1_p = np.zeros(xpoints)
        for j in range(xpoints):
            nu1_p_j = LIF0([mu_p_0[j], D_p_0[j], Cmp, gLp, Vre, EL, Nv, Vlb, Vth])
            nu1_p[j] = nu1_p_j

        nu1_s = np.zeros(xpoints)
        for j in range(xpoints):
            nu1_s_j = LIF0([mu_s_0[j], D_s_0[j], Cms, gLs, Vre, EL, Nv, Vlb, Vth])
            nu1_s[j] = nu1_s_j
            
        nu1_v = np.zeros(xpoints)
        for j in range(xpoints):
            nu1_v_j = LIF0([mu_v_0[j], D_v_0[j], Cmv, gLv, Vre, EL, Nv, Vlb, Vth])
            nu1_v[j] = nu1_v_j
            
        nancase = False
        if np.any(np.isnan(nu1_e)):
            print('WARNING: nan in nu1_e')
            nu1_e[np.isnan(nu1_e)] = 0
            nancase = True
        if np.any(np.isnan(nu1_p)):
            print('WARNING: nan in nu1_p')
            nu1_p[np.isnan(nu1_p)] = 0
            nancase = True
        if np.any(np.isnan(nu1_s)):
            print('WARNING: nan in nu1_s')
            nu1_s[np.isnan(nu1_s)] = 0
            nancase = True
        if np.any(np.isnan(nu1_v)):
            print('WARNING: nan in nu1_v')
            nu1_v[np.isnan(nu1_v)] = 0
            nancase = True
            
#         if nancase:
#             return nu_e, nu_p, nu_s, arrs, status
        print(nu1_e)
        print(nu1_p)
        print(nu1_s)
        print(nu1_v)
    
            
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

print('saving to: ', parname)
print('==== N = %s ===='%N)

param = np.load('mf_results/%s/pars.npy'%parname)

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
j_mat = np.array([[j_ee, j_ep, j_es, j_ev], [ j_pe, j_pp, j_ps, j_pv], [j_se, j_sp, j_ss, j_sv], [j_ve, j_vp, j_vs, j_vv]])
q_mat = np.array([[q_e, q_p, q_s, q_v] for _ in range(4)])


# connection probabilities
kbar_mat = np.array([[kbar_ee, kbar_ep, kbar_es, kbar_ev], [kbar_pe, kbar_pp, kbar_ps, kbar_pv],
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


nu_e = np.array([0.1 for i in range(200)])
nu_p = np.array([0.1 for i in range(200)])
nu_s = np.array([0.1 for i in range(200)])
nu_v = np.array([0.1 for i in range(200)])

nu_e, nu_p, nu_s, nu_v, arrs, status = find_fp_4pop([nu_e, nu_p, nu_s, nu_v], [j_e, j_p, j_s, j_v], max_step=1000, N=N, qq=0.05, tolerance = 1e-4)

if status == 0:
    print('converged')
    with open(f'meanfield_results/%s/N_%s.pkl'%(parname, N),'wb') as f:
        pickle.dump([x, nu_e, nu_p, nu_s, nu_v], f)
else:
    print('no convergence')
    with open(f'meanfield_results/%s/N_%s_noconv.pkl'%(parname, N),'wb') as f:
        pickle.dump([x, nu_e, nu_p, nu_s, nu_v], f)
