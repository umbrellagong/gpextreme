import sys
sys.path.append("../")
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from core import *
from sir import f
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from joblib import Parallel, delayed


def main(n_dim=2, num_grids_mc=201, n_init=4, n_seq=96, num_cores=20,
         input_type='Gaussian', w_n=None, sigma_n=None):

    num_seeds = 100
    domain = [[-6, 6] for i in range(n_dim)]
        
    if input_type == 'Gaussian':
        if w_n is None:
            w_n = np.zeros(n_dim)
        if sigma_n is None:
            sigma_n = np.eye(n_dim)
        inputs = GaussianInputs(w_n, sigma_n, domain)
    elif input_type == 'Uniform':
        inputs = UniformInputs(domain)

    # Generate 2D MC samples 
    bw = 0.1
    grids = [np.linspace(domain[i][0],  domain[i][1],  num_grids_mc) 
                for i in range(n_dim)]
    meshes = np.meshgrid(*grids)
    grids = np.concatenate([mesh.reshape(-1, 1) for mesh in meshes], axis=1)
    pdf_grids = inputs.pdf(grids)

    # Generate function_true
    filename = ('true_funcs/' + str(n_dim) + 'd.npy')
    if os.path.exists(filename):
        f_grids = np.load(filename, allow_pickle=True)
    else:
        f_grids = f(grids, num_cores)
        np.save(filename, f_grids)
    kde_t = custom_KDE(f_grids, pdf_grids, bw) 
    # define kernel   
    kernel = C(0.25, 'fixed') * RBF((0.5, 10), 'fixed')
                                
    # seed is for initial design, f_num is for function
    def wrapper(seed):    
        res_list = []

        cases = (((0.6, 0.6), (0, 0)),
                 ((0.8, 0.8), (0, 0)),
                 ((0.9, 0.9), (0, 0)),
                 ((1.0, 1.0), (0, 0)),
                 ((1.1, 1.1), (0, 0)),
                 ((1.2, 1.2), (0, 0)),
                 ((1.4, 1.4), (0, 0)),
                 ((1.6, 1.6), (0, 0)),
                 ((1.8, 1.8), (0, 0)),
                 ((2, 2), (0, 0))
                 )
        
        for case in cases:
            np.random.seed(seed)
            res_list.append(OptimalDesign(f, inputs, kde_t, 
                                          n_init, n_seq,
                                          kernel, bw, grids,
                                          case[0], case[1]))

        return res_list

    res_list = Parallel(n_jobs=num_cores)(delayed(wrapper)(j) 
                                            for j in range(num_seeds))
    np.save('results/res_t', np.array(res_list, dtype=object))
    
if __name__ == '__main__':
    main(n_dim=2, n_seq=46, num_grids_mc=121, num_cores=10)