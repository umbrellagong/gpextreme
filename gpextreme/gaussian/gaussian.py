import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor


def generate_true_funcs(num_funcs, domain, num_grids, kernel, 
                        samples_load_address=None, index_start=0, 
                        save_address=None, whether_itp=True):
    n_dim = len(domain)
    grid_list = [np.linspace(domain[i][0],  domain[i][1],  num_grids) 
                    for i in range(n_dim)]
    meshes = np.meshgrid(*grid_list)
    
    if samples_load_address:
        samples_list = np.load(samples_load_address, 
                             allow_pickle=True)[index_start:index_start+num_funcs]
    else:
        grids = np.concatenate([mesh.reshape(-1, 1) for mesh in meshes], axis=1)
        model = GaussianProcessRegressor(kernel)
        samples_list = model.sample_y(grids, n_samples=num_funcs, 
                                      random_state=0).T

    if whether_itp:
        res_list = [RegularGridInterpolator(grid_list, 
                                            samples.reshape(meshes[0].shape), 
                                            method='cubic') 
                        for samples in samples_list]
    else:
        res_list = samples_list
    if save_address:
        np.save(save_address, res_list)    
    return res_list