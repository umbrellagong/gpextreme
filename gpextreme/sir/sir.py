import numpy as np
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

multiplier = 3*10**-9 # Special for the map

class Noise:

    def __init__(self, domain, sigma=0.1, ell=1.0):
        self.ti = domain[0]
        self.tf = domain[1]
        self.tl = domain[1] - domain[0]
        self.R = self.get_covariance(sigma, ell)    
        self.lam, self.phi = self.kle(self.R)

    def get_covariance(self, sigma, ell):
        m = 500 + 1
        self.t = np.linspace(self.ti, self.tf, m)
        self.dt = self.tl/(m-1)
        R = np.zeros([m, m])
        for i in range(m):
            for j in range(m):
                tau = self.t[j] - self.t[i]
                R[i,j] = sigma*np.exp(-tau**2/(2*ell**2)) 
        return R*self.dt

    def kle(self, R):
        lam, phi = np.linalg.eigh(R)
        phi = phi/np.sqrt(self.dt)
        idx = lam.argsort()[::-1]
        lam = lam[idx]
        phi = phi[:,idx]
        return lam, phi

    def get_eigenvalues(self, trunc=None):
        return self.lam[0:trunc]

    def get_eigenvectors(self, trunc=None):
        return self.phi[:,0:trunc]

    def get_sample(self, xi):
        nRV = np.asarray(xi).shape[0]
        phi_trunc = self.phi[:,0:nRV]
        lam_trunc = self.lam[0:nRV]
        lam_sqrtm = np.diag(np.sqrt(lam_trunc))
        sample = np.dot(phi_trunc, np.dot(lam_sqrtm, xi))
        return sample

    def get_sample_interp(self, xi):
        sample = self.get_sample(xi.ravel())
        sample_int = interp1d(self.t, sample, kind='cubic')
        return sample_int  # 然后我去inter就好了嘛. 


def map_def(theta, gamma, delta, N, I0, T, dt): 
    beta_itp = noise.get_sample_interp(theta)
    beta = multiplier * (beta_itp(np.linspace(0, 1, (int(T/dt)))) + 2.55)
    S = np.zeros((int(T/dt),))
    S[0] = N
    I= np.zeros((int(T/dt),))
    I[0] = I0
    R = np.zeros((int(T/dt),)) # 这个很简单啊，一开始就是0而已啊. 
    for tt in range(0,np.size(S)-1):
        # Ordinary different equations of the model
        dS = (-beta[tt]*I[tt]*S[tt] + delta*R[tt]) * dt
        dI = (beta[tt]*I[tt]*S[tt] - gamma*I[tt]) * dt
        dR = (gamma*I[tt] - delta*R[tt]) * dt
        # Simple integration
        S[tt+1] = S[tt] + dS
        I[tt+1] = I[tt] + dI
        R[tt+1] = R[tt] + dR
    return I[-1] / 1e7

# define parameter
T = 45  
dt = 0.1
gamma = 0.25 
delta = 0
N_people = 10*10**7
I0 = 50


noise = Noise([0,1], sigma=0.1, ell=1)

def f(X, num_cores=1):
    if X.ndim==1:
        return map_def(X, gamma, delta, N_people, I0, T, dt)
    if X.ndim==2:
        if num_cores==1:
            return np.array([map_def(x, gamma, delta, N_people, I0, T, dt) 
                             for x in X])
        else:
            return np.array(Parallel(n_jobs=num_cores)(delayed(map_def)(x, 
                            gamma, delta, N_people, I0, T, dt) for x in X))

def compute_truth(n_dim=2, num_grids_mc=121, num_cores=1, address=None):
    domain = [[-6, 6] for i in range(n_dim)]
    grids = [np.linspace(domain[i][0],  domain[i][1],  num_grids_mc) 
                for i in range(n_dim)]
    meshes = np.meshgrid(*grids)
    grids = np.concatenate([mesh.reshape(-1, 1) for mesh in meshes], axis=1)

    f_grids = f(grids, num_cores)
    np.save(address, f_grids)

if __name__=='__main__':
    compute_truth(num_grids_mc=121, num_cores=10, address='ture_func')