import numpy as np
from matplotlib import pyplot as plt

class cdf_sampler(object):
    
    def __init__(self,x,y):
        self.x_input  = x
        self.freq_d   = y

        pdf_fnorm = np.sum(y)
        self.cdf       = np.zeros(len(y))
        val       = 0.0
        for i in range(len(self.cdf)):
            val+=y[i]
            self.cdf[i] = val/pdf_fnorm

    def sample_n(self,n):

        self.sample = np.zeros(n)
        for i in range(n):
            tm = np.random.uniform()
            tt = np.where(np.abs(tm-self.cdf) == np.min(np.abs(tm-self.cdf)))

            self.sample[i] = self.x_input[tt[0]]
