import numpy as np
from matplotlib import pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from scipy import integrate as integ

import TAOIST_MC as tao


def do_it():
    zem = 2.5
    dz = 5e-5
    zs = np.arange(0,zem+.01,dz)
    NHIs = np.arange(12.,21.,.1)
    wav = np.arange(600.*(1.+zem),1500.*(1.+zem),10.)
    nex = 25

    F  = plt.figure()
    ax = F.add_subplot(111)
    taua= np.zeros((nex,len(wav)))
    for n in range(nex):
        
        tm_fzs = tao.make_zdist(zs,dz)
        tm_tau = tao.make_tau(zs,dz,tm_fzs,NHIs,wav)

        ax.plot(wav,np.exp((-1.)*tm_tau),'k-',lw=.3)
        taua[n,:] = tm_tau

    taum = np.mean(np.exp((-1.)*taua),axis=0)

    ax.plot(wav,taum,'g-',lw=2)
    ax.axvline(x=911.75*(1.+zem),c='r',ls='--')
    ax.axvline(x=1216.*(1.+zem),c='r',ls='--')
    ax.set_ylim(-.1,1.1)
    plt.show()

do_it()
