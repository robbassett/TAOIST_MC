import numpy as np
from matplotlib import pyplot as plt

class cdf_sampler(object):

    # __init__ computed the normalisation
    # factor then constructs the cumulative
    # distribution function (cdf). During this 
    # step y is checked for zeros, and these 
    # are replaced with 1/50th of the minumum
    # value in y. This is done so that the cdf
    # is single valued, and the 1/50th ensures
    # that the values associated with zeros in
    # in the pdf have a very small probability
    # of being selected.
    
    def __init__(self,x,y):
        self.x_input  = x
        self.freq_d   = y

        pdf_fnorm = np.sum(y)
        t = np.where(y == 0.)
        g = np.where(y != 0.)
        mv= np.min(y[g[0]])/50.
        
        pdf_fnorm+=float(len(t[0]))*mv
        
        self.cdf       = np.zeros(len(y))
        val       = 0.0
        for i in range(len(self.cdf)):
            val+=y[i]
            if y[i] == 0.:
                val+=mv
            
            
            self.cdf[i] = val/pdf_fnorm

    # sample_n in produces a random sample
    # of n with a distribution matched to the
    # input array, y.
    
    def sample_n(self,n):

        self.sample = np.zeros(n)
        for i in range(n):
            tm = np.random.uniform()
            tt = np.where(np.abs(tm-self.cdf) == np.min(np.abs(tm-self.cdf)))

            self.sample[i] = self.x_input[tt[0]]
