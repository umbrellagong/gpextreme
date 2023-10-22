import numpy as np
import scipy.stats as stats
from pyDOE import lhs


class GaussianInputs: 
    'A class for Gaussian inputs'
    def __init__(self, mean, cov, domain):
        self.mean = mean
        self.cov = cov
        self.domain = domain
        self.dim = len(mean)
        
    def sampling(self, num, criterion=None):
        if criterion == 'random':
            return np.random.multivariate_normal(self.mean, self.cov, num)
        else:
            lhd = lhs(self.dim, num, criterion=criterion)
            lhd = self.rescale_samples(lhd, self.domain)
            return lhd
            
    def pdf(self, x):
        return stats.multivariate_normal(self.mean, self.cov).pdf(x) 
    
    @staticmethod
    def rescale_samples(x, domain):
        """Rescale samples from [0,1]^d to actual domain."""
        for i in range(x.shape[1]):
            bd = domain[i]
            x[:,i] = x[:,i]*(bd[1]-bd[0]) + bd[0]
        return x


class UniformInputs: 
    'A class for Gaussian inputs'
    def __init__(self, domain):
        self.domain = domain
        self.dim = len(domain)
        
    def sampling(self, num, criterion=None):
        lhd = lhs(self.dim, num, criterion=criterion)
        lhd = self.rescale_samples(lhd, self.domain)
        return lhd
            
    def pdf(self, x):
        return np.ones(len(x))          

    @staticmethod
    def rescale_samples(x, domain):
        """Rescale samples from [0,1]^d to actual domain."""
        for i in range(x.shape[1]):
            bd = domain[i]
            x[:,i] = x[:,i]*(bd[1]-bd[0]) + bd[0]
        return x