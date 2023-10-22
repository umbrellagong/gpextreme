import sys
sys.path.append("../")
import os

import numpy as np
from core import *
from gaussian import generate_true_funcs
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from joblib import Parallel, delayed


def main(n_dim=2, num_grids_itp=101, num_grids_mc=201, 
         n_init=4, n_seq=146, num_funcs=200,     
         num_cores=20, num_seeds=20,
         kernel_type='Matern', tau_n=None, l_n=None, nu=1.5,
         input_type='Gaussian', w_n=None, sigma_n=None,
         t=1.0, whether_load_res=False):


    domain = [[-5, 5] for i in range(n_dim)]

    # specify kernel and input 
    if tau_n is None:
        tau_n = 4
    if l_n is None:
        l_n = np.array([1] * n_dim)

    
    if kernel_type == 'Matern':
        kernel = C(tau_n, 'fixed') * Matern(l_n, 'fixed', nu=nu)
    elif kernel_type == 'RBF':
        kernel = C(tau_n, 'fixed') * RBF(l_n, 'fixed')
        
    if input_type == 'Gaussian':
        if w_n is None:
            w_n = np.zeros(n_dim)
        if sigma_n is None:
            sigma_n = np.eye(n_dim)
        inputs = GaussianInputs(w_n, sigma_n, domain)
    elif input_type == 'Uniform':
        inputs = UniformInputs(domain)
    
    # Generate function_true
    filename = ('true_funcs/' + str(n_dim) + 'd_' 
                + kernel_type + '_' 
                + str(tau_n) + ''.join([str(l).replace('.','') for l in l_n])
                + '.npy')
    if os.path.exists(filename):
        f_true_list = np.load(filename, allow_pickle=True)
    else:
        f_true_list = generate_true_funcs(num_funcs, domain, num_grids_itp, kernel, 
                                          samples_load_address=None, 
                                          index_start=0, save_address=filename, 
                                          whether_itp=True)
 
    # Generate 2D MC samples 
    bw = 0.1
    grids = [np.linspace(domain[i][0],  domain[i][1],  num_grids_mc) 
                for i in range(n_dim)]
    meshes = np.meshgrid(*grids)
    grids = np.concatenate([mesh.reshape(-1, 1) for mesh in meshes], axis=1)
    pdf_grids = inputs.pdf(grids)
    kde_t_list = Parallel(n_jobs=num_cores)(delayed(custom_KDE)(f_true_list[j](grids), 
                                                                pdf_grids, bw) 
                                            for j in range(num_funcs))

    
    save_address = ('results/t' 
                     + str(t)[0] + str(t)[2] + '/'
                     + str(n_dim) + 'd_' 
                     + kernel_type + '_' 
                     + str(tau_n) + ''.join([str(l).replace('.','') for l in l_n]) + '_'
                     + str(input_type))
    
    if whether_load_res:
        prev_address = save_address + '_prev.npy'
        prev_res = np.load(prev_address, allow_pickle=True)

    
    # seed is for initial design, f_num is for function
    def wrapper(seed, f_num):    
        res_list = []
        cases = (   ((t, t), (0, 0)),
                    ((t, t), (1, 1)),
                    ((t, t), (2, 2)),
                    ((t, t), (3, 3)),
                    ((t, t), (4, 4)),
                    ((t, t), (6, 6)),
                    ((t, t), (8, 8)))
        
        for i, case in enumerate(cases):
            np.random.seed(seed)
            if whether_load_res:
                load_res = prev_res[f_num][seed][i]
            else:
                load_res = None
            res_list.append(OptimalDesign(f_true_list[f_num], inputs, 
                                        kde_t_list[f_num], n_init, n_seq, 
                                        kernel, bw, grids, 
                                        case[0], case[1], 
                                        load_res=load_res))
        return res_list

    res_list_list = []

    for f_num in range(num_funcs):
        res_list = Parallel(n_jobs=num_cores)(delayed(wrapper)(j, f_num) 
                                            for j in range(num_seeds))
        res_list_list.append(res_list)
    np.save(save_address, np.array(res_list_list, dtype=object))
    
if __name__ == '__main__':
    for t in (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4):
        main(n_dim=2, kernel_type='RBF', input_type='Gaussian', t=t, 
             n_seq=146)