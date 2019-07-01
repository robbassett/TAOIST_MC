import numpy as np
from matplotlib import pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from scipy import integrate as integ
import cdf_sampler as cds


def dX(z,Om0,Ode0):
    num = ((1.+z)**2.)
    den = np.sqrt(Ode0+(Om0*((1.+z)**3)))
    return num/den

def dZ_2_dX(z,dz,Om0,Ode0):
    tDx,err = integ.quad(dX,z,z+dz,args=(Om0,Ode0))
    return tDx

def N_abs(NHI,dNHI,z,dX):
    if NHI >= 12.0 and NHI < 15.2:
        beta  = 1.635
        A     = 10.**9.305
        gamma = 2.5

    if NHI >= 15.2:
        beta  = 1.463
        A     = 10.**7.542
        gamma = 1.0

        
    c1 = (10.**NHI)**((-1.)*beta)
    c2 = A*((1.+z)**gamma)
    c3 = dNHI*dX
    
    return c1*c2*c3


def N_single_z(NHIs,z,dX,n):
    Nabs = np.zeros(len(NHIs)-1)
    dns  = np.zeros(len(Nabs))
    arr  = np.array([])
    
    for i in range(len(Nabs)):
        dns[i] = (10.**NHIs[i+1])-(10.**NHIs[i])
        Nab = N_abs(NHIs[i],dns[i],z,dX)
        Nabs[i] = np.random.poisson(Nab)
        tm = np.zeros(int(Nabs[i]))+NHIs[i]
        arr = np.concatenate([arr,tm])

    choi = 0.0
    if len(arr) > 0:
        if len(arr) > n:
            ii = np.random.randint(0,len(arr),size=n)
            choi=arr[ii]
        else:
            choi=np.array(arr)
    return Nabs,dns,choi

def fz_HI(z,dz):
    A = 400.
    z1,z2 = 1.2,4.0
    g1,g2,g3 = .2,2.5,4.0

    if z <= z1:
        c = ((1.+z)/(1.+z1))**g1
    if z > z1 and z <= z2:
        c = ((1.+z)/(1.+z1))**g2
    if z > z2:
        c1 = ((1.+z2)/(1.+z1))**g2
        c2 = ((1.+z)/(1.+z2))**g3
        c  = c1*c2

    return A*c*dz

def make_zdist(zs,dz):
    fzs = np.zeros(len(zs))
    for i in range(len(zs)):
        fzs[i] = np.random.poisson(fz_HI(zs[i],dz))
    return fzs

def tau_HI_LyC(NHI,lam,z):
    l_lc = 911.8*(1.+z)
    
    x = lam/l_lc
    tau = NHI*(6.3e-18)*(x*x*x)

    t = (lam/l_lc > 1.)
    t = np.where(t == True)
    tau[t[0]] = 0.

    return tau

# voigt profile approximation from Tepper-Garcia 2006
def voigt_approx(lam,lami,b,gamma):
    c = 2.998e18 # angst/s
    ldl = (b/c)*lami
    a = ((lami*lami)*gamma)/(4.*np.pi*c*ldl)
    x = (lam-lami)/ldl

    A1 = np.exp((-1.)*x*x)
    A2= a*(2./np.sqrt(np.pi))

    K1 = (1./(2.*x*x))
    K2 = ((4.*x*x)+3.)*((x*x)+1.)*A1
    K3 = (1./(x*x))*((2.*x*x)+3.)*(np.sinh(x*x))

    Kx = K1*(K2-K3)

    return A1*(1.-(A2*Kx))

def doppler_dist(b):
    bs = 23.
    A1 = (4.*bs*bs*bs*bs)/(b*b*b*b*b)
    A2 = np.exp((-1.)*A1*b/4.)
    return A1*A2*1.e13

def tau_HI_LAF(wav,z):
    me,ce,c = 9.1094e-31,1.6022e-19,2.99792e18
    LAF_table = np.loadtxt('./Lyman_series.dat',float)

    tau = np.zeros(len(wav))
    lam = wav/(1.+z)

    bx = np.arange(1,1000,.1)
    by = doppler_dist(bx)
    bcds  =  cds.cdf_sampler(bx,by)
    bcds.sample_n(1)
    
    b  = bcds.sample[0]*1.e13
    print(b)
    for i in range(len(LAF_table[:,0])):
        sig_T = 6.625e-25       #cm^2
        c     = 2.998e10        #cm/s
        fi    = LAF_table[i,1]  
        li    = LAF_table[i,0]  #angstrom
        gamma = LAF_table[i,2]

        A1 = c*np.sqrt((3.*np.pi*sig_T)/8.)
        A2 = (fi*li)/(np.sqrt(np.pi)*b)
        A3 = voigt_approx(lam,li,b,gamma)

        tm_tau = A1*A2*A3
        bad = np.where(np.isfinite(tm_tau) == False)
        tm_tau[bad[0]] = 0.
        tau+=tm_tau
        
    return tau

def make_tau(zs,dz,fzs,NHIs,wav):

    tau = np.zeros(len(wav))
    for i in range(len(zs)):
        if fzs[i] != 0.:
            DX = dZ_2_dX(zs[i],zs[i]+dz,cosmo.Om0,cosmo.Ode0)
            Nabs,dn,cdt = N_single_z(NHIs,zs[i],DX,int(fzs[i]))

            cdt = 10.**cdt
            cdt = np.sum(cdt)
            tau+=tau_HI_LyC(cdt,wav,zs[i])
            tau+=cdt*tau_HI_LAF(wav,zs[i])
            
    return tau










