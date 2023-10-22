import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from .utility import * 
from .gpr import *

def OptimalDesign(f, inputs, true, n_init, n_seq, 
                  kernel, bw, grids, 
                  bound_power, bound_alpha, 
                  known_pdf_y=False, load_res=None):
    
    pdf_grids = inputs.pdf(grids)
    error = []
    sgp = GaussianProcessRegressor(kernel, normalize_y=False, 
                           n_restarts_optimizer=10) 
    # initial samples
    if load_res is None: 
        DX = inputs.sampling(n_init)
    else:
        DX = load_res[0]
        n_init = len(DX)

    DY = f(DX)
    sgp.fit(DX, DY)
    mean_grids, std_grids = sgp.predict(grids, return_std=True)
    var_grids = std_grids ** 2 # that will keep same...
    
    # firstly get the initial things... 

    # kernel is the most time-consuming part, so memorize it. 
    K_cache = np.zeros((n_init + n_seq, len(grids)))
    K_cache[:n_init] = sgp.kernel_(DX, grids) 
    
    # sequential samples
    for i in range(n_seq):
        # compute error 
        error.append(log_pdf([sgp], true, grids, pdf_grids, bw, [mean_grids])[0])
        # generate weight for each grid value

        w_grids = np.zeros(len(mean_grids))
        if bound_alpha == (0, 0):
            pred_grids_list = ((mean_grids, 1),) 
        else:
            alpha = bound_alpha[0] + i * (bound_alpha[1] 
                                        - bound_alpha[0]) / (n_seq)
            pred_grids_list = ((mean_grids, 1), 
                            (mean_grids + alpha * std_grids, 1), 
                            (mean_grids - alpha * std_grids, 1))
            
        for pred_grids, weight in pred_grids_list:
            if known_pdf_y:
                x_kde, y_kde = true.evaluate()
            else:
                x_kde, y_kde = custom_KDE(pred_grids, pdf_grids, bw).evaluate()
            
            pdf_func = InterpolatedUnivariateSpline(x_kde, y_kde)
            pdf_y_grids = np.clip(pdf_func(np.clip(pred_grids, x_kde[0], 
                                                   x_kde[-1])), 1e-16, None)
            
            power = bound_power[0] + i * (bound_power[1] - 
                                            bound_power[0]) / (n_seq)
            w_grids += weight * (pdf_grids / pdf_y_grids ** power)


        x_opt = np.atleast_2d(grids[np.argmax(var_grids * w_grids)])
        y_opt = f(x_opt)[0]

        DX = np.append(DX, x_opt, axis=0)
        DY = np.append(DY, y_opt)
        mean_grids, var_grids = sgp.predict_grids_fast(DX[-1], DY[-1], 
                                        grids, K_cache, mean_grids, var_grids)
        sgp.fit(DX, DY)
        std_grids = np.sqrt(np.clip(var_grids, 1e-10, None))
    error.append(log_pdf([sgp], true, grids, pdf_grids, bw, [mean_grids])[0])
    return DX, error


def OptimalDesign_hyper(f, inputs, true, n_init, n_seq, 
                        kernel, bw, grids, 
                        bound_power, bound_alpha, 
                        known_pdf_y=False, load_res=None,
                        opt_hyper_threshold=20, opt_hyper_interval=10,
                        cut_off=(-np.inf, np.inf)):
    
    pdf_grids = inputs.pdf(grids)
    error = []
    sgp = GaussianProcessRegressor(kernel, normalize_y=False, 
                           n_restarts_optimizer=10) 

    # initial samples
    if load_res is None: 
        DX = inputs.sampling(n_init)
    else:
        DX = load_res[0]
        n_init = len(DX)

    DY = f(DX)
    sgp.fit(DX, DY)
    mean_grids, std_grids = sgp.predict(grids, return_std=True)
    var_grids = std_grids ** 2
    
    if opt_hyper_threshold <= n_seq:
        K_cache = np.zeros((n_init + n_seq, len(grids)))
        if opt_hyper_threshold == 0:
            K_cache[:n_init] = sgp.kernel_(DX, grids) 

            
    
    # sequential samples
    for i in range(n_seq):
        # compute error 
        error.append(log_pdf([sgp], true, grids, pdf_grids, bw, 
                     [mean_grids], cut_off)[0])
        # Generate weight for each grid value
        
        w_grids = np.zeros(len(mean_grids))
        if bound_alpha == (0, 0):
            pred_grids_list = ((mean_grids, 1),) 
        else:
            alpha = bound_alpha[0] + i * (bound_alpha[1] 
                                        - bound_alpha[0]) / (n_seq)
            pred_grids_list = ((mean_grids, 1), 
                            (mean_grids + alpha * std_grids, 1),
                            (mean_grids - alpha * std_grids, 1))
            
        for pred_grids, weight in pred_grids_list:
            if known_pdf_y:
                x_kde, y_kde = true.evaluate()
            else:
                x_kde, y_kde = custom_KDE(pred_grids, pdf_grids, bw).evaluate()
            
            pdf_func = InterpolatedUnivariateSpline(x_kde, y_kde)
            pred_grids = np.where(pred_grids < cut_off[0], cut_off[0], pred_grids)
            pred_grids = np.where(pred_grids > cut_off[1], cut_off[1], pred_grids)
            
            pdf_y_grids = np.clip(pdf_func(np.clip(pred_grids, x_kde[0], 
                                                   x_kde[-1])), 1e-16, None)
            
            power = bound_power[0] + i * (bound_power[1] - 
                                            bound_power[0]) / (n_seq)
            w_grids += weight * (pdf_grids / pdf_y_grids ** power)


        x_opt = np.atleast_2d(grids[np.argmax(var_grids * w_grids)])
        y_opt = f(x_opt)[0]

        DX = np.append(DX, x_opt, axis=0)
        DY = np.append(DY, y_opt)
        
        if ((i + 1) <= opt_hyper_threshold or 
            ((i + 1) > opt_hyper_threshold and 
             (i + 1 - opt_hyper_threshold) % opt_hyper_interval == 0)):
            sgp.fit(DX, DY)
            mean_grids, std_grids = sgp.predict(grids, return_std=True)
            var_grids = std_grids ** 2
            if i >= opt_hyper_threshold:
                K_cache[:n_init + i + 1] = sgp.kernel_(DX, grids)
        else:
            mean_grids, var_grids = sgp.predict_grids_fast(DX[-1], DY[-1], 
                                        grids, K_cache, mean_grids, var_grids)
            sgp.fit_keep(DX, DY)
            std_grids = np.sqrt(np.clip(var_grids, 1e-10, None))
    
    error.append(log_pdf([sgp], true, grids, pdf_grids, bw, [mean_grids], cut_off)[0])
    return DX, error