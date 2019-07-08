import numpy as np
from matplotlib import pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from scipy import integrate as integ
import cdf_sampler as cds

# - - - - - - - - - - - - - - - - - - - - - - - - -
# conversion from dz to dX, comoving coordinates
# e.g. Steidel et al. 2018, Appendix B, eq 25.
# - - - - - - - - - - - - - - - - - - - - - - - - -
# Here we are assuming WMAP9 cosmology from
# astropy. To change, update "from astropy.comology..."
# above or manually enter omega matter and omega lambda
# in function "make_tau" below.
# - - - - - - - - - - - - - - - - - - - - - - - - -
# z    = redshift
# Om0  = omega matter
# Ode0 = omega lambda
# - - - - - - - - - - - - - - - - - - - - - - - - -
def dX(z,Om0,Ode0):
    num = ((1.+z)**2.)
    den = np.sqrt(Ode0+(Om0*((1.+z)**3)))
    return num/den


# - - - - - - - - - - - - - - - - - - - - - - - - -
# Computing the comoving pathlength from redshift
# interval. Again assuming WMAP9 comsology, see above
# on info for using alternative cosmology.
# - - - - - - - - - - - - - - - - - - - - - - - - -
# z    = redshift
# dz   = redshift interval
# Om0  = omega matter
# Ode0 = omega lambda
# - - - - - - - - - - - - - - - - - - - - - - - - -
def dZ_2_dX(z,dz,Om0,Ode0):
    tDx,err = integ.quad(dX,z,z+dz,args=(Om0,Ode0))
    return tDx

# - - - - - - - - - - - - - - - - - - - - - - - - -
# N_abs computes the Poissonian event rate (lambda)
# for a given redshift and column density along with
# the associated redshift interval and column density
# bin sizes. Here we adopt the functional form of a
# broken power law following Table 11 of Steidel
# et al. 2018 (Appendix B).
# - - - - - - - - - - - - - - - - - - - - - - - - -
# NHI  = central value of current NHI bin
# dNHI = current NHI bin size
# z    = current redshift
# dX   = comoving path length computed using
#        above functions
# - - - - - - - - - - - - - - - - - - - - - - - - -
# returns Poissonian event rate for given bin. This
# value is used in "N_single_z" with np.random.poisson
# to give the random number of absorbers in a given
# NHI bin and given redshift bin.
# - - - - - - - - - - - - - - - - - - - - - - - - -
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

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute the number of absorbers in bins of NHI at
# in a given redshift bin based on Poisson sampling
# of the HI column density distribution function,
# then randomly chooses n absorbers where n is
# is determined by sampling the redshift distribution
# of absorption systems (see "fz_HI"). 
# - - - - - - - - - - - - - - - - - - - - - - - - -
# TO DO!! replace this with the inverse CDF sampler
# - - - - - - - - - - - - - - - - - - - - - - - - -
# NHIs = array of log10(NHIs) for bins
# z    = current redshift
# dX   = comoving path length of redshift bin
# n    = number of absorbers in current reshift bin
#        determined by poisson sampling of fz_HI
# - - - - - - - - - - - - - - - - - - - - - - - - -
# returns:
# Nabs = full random sample of absorbers, array of
#        length len(NHIs)-1 where each value indicates
#        number of absorbers in a given NHI bin
# dns  = array of NHI bin sizes in linear space (NHI
#        bins are input in log space**)
# choi = array of n randomly selected absorber NHI
#        values
# - - - - - - - - - - - - - - - - - - - - - - - - -
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

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Redshift HI absorption system distribution function
# taken from Inoue & Iwata 2008, eq 5. Returns the
# Poissonian lambda value to predict the number of
# absorbers at all NHI in a given redshift bin.
# - - - - - - - - - - - - - - - - - - - - - - - - -
# z  = current redshift
# dz = redshift bin width
# - - - - - - - - - - - - - - - - - - - - - - - - -
# returns Poissonian lambda to be used in "make_zdist"
# below.
# - - - - - - - - - - - - - - - - - - - - - - - - -
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

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Gives the number of absorbers in each redshift
# bin based on Poisson sampling of fz_HI
# - - - - - - - - - - - - - - - - - - - - - - - - -
# TO DO: replace with inverse cdf sampler
# - - - - - - - - - - - - - - - - - - - - - - - - -
# zs = array of redshift bin values
# dz = redshift bin size
# - - - - - - - - - - - - - - - - - - - - - - - - -
# returns array of number of absorbers in each
# redshift bin
# - - - - - - - - - - - - - - - - - - - - - - - - -
def make_zdist(zs,dz):
    fzs = np.zeros(len(zs))
    for i in range(len(zs)):
        fzs[i] = np.random.poisson(fz_HI(zs[i],dz))
    return fzs

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Computes the optical depth for LyC photons for
# an absorber of column density NHI at redshift z.
# - - - - - - - - - - - - - - - - - - - - - - - - -
# NHI = hydrogen column density
# lam = wavelength array
# z   = redshift of absorber
# - - - - - - - - - - - - - - - - - - - - - - - - -
# returns array with length len(lam) of optical
# depth (tau) values at each wavelength. Convert to
# transmission with "np.exp((-1.)*tau)"
# - - - - - - - - - - - - - - - - - - - - - - - - -
def tau_HI_LyC(NHI,lam,z):
    l_lc = 911.8*(1.+z)
    
    x = lam/l_lc
    tau = NHI*(6.3e-18)*(x*x*x)

    t = (lam/l_lc > 1.)
    t = np.where(t == True)
    tau[t[0]] = 0.

    return tau

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Voigt profile approximation from Tepper-Garcia 2006.
# Used to compute the Ly-alpha forest transmission
# with "tau_HI_LAF" below.
# - - - - - - - - - - - - - - - - - - - - - - - - -
# lam   = wavelength array
# lami  = central wavelength of current Lyman line
# b     = doppler broadening in angst/s
# gamma = damping parameter of current Lyman line
#         taken from VPFIT table
# - - - - - - - - - - - - - - - - - - - - - - - - -
# returns array with voigt profile for current
# lyman line with length equal to len(lam)
# - - - - - - - - - - - - - - - - - - - - - - - - -
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

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Doppler parameter distribution function taken from
# Inoue & Iwata 2008, eq 6. Used to randomly sample
# a doppler broadening for a given absorber. Sampling
# is done using inverse cdf sampling with
# cdf_sampler.py
# - - - - - - - - - - - - - - - - - - - - - - - - -
# b = doppler broadening
# - - - - - - - - - - - - - - - - - - - - - - - - -
def doppler_dist(b):
    bs = 23.
    A1 = (4.*bs*bs*bs*bs)/(b*b*b*b*b)
    A2 = np.exp((-1.)*A1*b/4.)
    return A1*A2*1.e13

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Computes the Lyman line forest cross-sections as
# a function of wavelength following Inoue & Iwata
# 2008 eq 10. To convert to an optical depth spectrum,
# multiply by the column density of the current
# absorber as in "make_tau".
# - - - - - - - - - - - - - - - - - - - - - - - - -
# wav = wavelength array (angstroms)
# z   = redshift
# - - - - - - - - - - - - - - - - - - - - - - - - -
# returns cross section spectrum for Lyman lines
# - - - - - - - - - - - - - - - - - - - - - - - - -
def tau_HI_LAF(wav,z,LAF_table):
    me,ce,c = 9.1094e-31,1.6022e-19,2.99792e18
    

    tau = np.zeros(len(wav))
    lam = wav/(1.+z)

    bx = np.arange(1,1000,.1)
    by = doppler_dist(bx)
    bcds  =  cds.cdf_sampler(bx,by)
    bcds.sample_n(1)
    
    b  = bcds.sample[0]*1.e13  #angstrom/s
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

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Create the optical depth spectrum for a given
# sightline. Conver to transmission using
# "np.exp((-1.)*tau)".
# - - - - - - - - - - - - - - - - - - - - - - - - -
# zs   = redshift array
# dz   = redshift bin size
# fzs  = number of absorber in each redshift bin
# NHIs = HI column density bins
# wav  = wavelength array
# - - - - - - - - - - - - - - - - - - - - - - - - -
# returns the optical depth spectrum including both
# LyC and Lyman series line absorption
# - - - - - - - - - - - - - - - - - - - - - - - - -
def make_tau(zs,dz,fzs,NHIs,wav):

    tau = np.zeros(len(wav))
    LAF_table = np.loadtxt('./Lyman_series.dat',float)
    for i in range(len(zs)):
        if fzs[i] != 0.:
            DX = dZ_2_dX(zs[i],zs[i]+dz,cosmo.Om0,cosmo.Ode0)
            Nabs,dn,cdt = N_single_z(NHIs,zs[i],DX,int(fzs[i]))

            cdt = 10.**cdt
            cdt = np.sum(cdt)
            tau+=tau_HI_LyC(cdt,wav,zs[i])
            tau+=cdt*tau_HI_LAF(wav,zs[i],LAF_table)
            
    return tau










