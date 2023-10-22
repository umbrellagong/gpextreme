import numpy as np
from scipy import stats
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


def custom_KDE(data, weights=None, bw=None):
    if bw is None:
        try:
            sc = stats.gaussian_kde(data, weights=weights)
            bw = np.sqrt(sc.covariance).flatten()[0]
        except:
            bw = 1.0
        if bw < 1e-8:
            bw = 1.0
    return FFTKDE(bw=bw).fit(data, weights)


def log_pdf(m_list, kde_t, pts, weights, bw, mu_list=None, 
            cut_off=(-np.inf, np.inf)):
    res = []
    for i, model in enumerate(m_list):
        if mu_list is None:
            mu = model.predict(pts)
        else:
            mu = mu_list[i]
        kde_c = custom_KDE(mu.flatten(), weights=weights, bw=bw)

        x_min = min(kde_c.data.min(), kde_t.data.min())
        x_max = max( kde_c.data.max(), kde_t.data.max() )
        rang = x_max-x_min
        x_eva = np.linspace(x_min - 0.01*rang,
                            x_max + 0.01*rang, 1024)        
        yc, yt = kde_c.evaluate(x_eva), kde_t.evaluate(x_eva)
        
        yc = np.where(x_eva < cut_off[0], 1, yc)
        yc = np.where(x_eva > cut_off[1], 1, yc)
        yt = np.where(x_eva < cut_off[0], 1, yt)
        yt = np.where(x_eva > cut_off[1], 1, yt)
                
        log_yc, log_yt = np.log10(yc), np.log10(yt)

        np.clip(log_yc, -14, None, out=log_yc)
        np.clip(log_yt, -14, None, out=log_yt)

        log_diff = np.abs(log_yc-log_yt)
        noInf = np.isfinite(log_diff)
        res.append(np.trapz(log_diff[noInf], x_eva[noInf]))
    return res


def plot_2d(res_list, f_true_list, inputs, bw, num_grids,
                       index_func, cases_list, index_init, num_init, num_seq_list, 
                       sgp, save_address=None):
    assert len(cases_list) == 2
    levels = np.arange(-8, 8.1, 1)
    
    x = np.linspace(-5, 5, num_grids)
    y = np.linspace(-5, 5, num_grids)
    xv, yv = np.meshgrid(x, y)
    grids = np.concatenate((xv.reshape(-1, 1), yv.reshape(-1, 1)), axis=1)
    pdf_grids = inputs.pdf(grids)  
    
    f_grids = f_true_list[index_func](grids)
    kde_t = custom_KDE(f_grids, pdf_grids, bw)   
    max_pos, min_pos = grids[np.argmax(f_grids)], grids[np.argmin(f_grids)]
    print(max_pos, min_pos)
    
    fig, axes = plt.subplots(len(num_seq_list), 4, 
                             figsize=(13, 3 * len(num_seq_list)))
    color_list = ['tab:blue', 'tab:orange']

    for k in range(len(num_seq_list)):
        # figure 1 true function    
        axes[k][0].contourf(xv, yv, f_grids.reshape(xv.shape), levels=levels)
        
        # figure 2/3 pred functions         
        num_seq = num_seq_list[k]
        for i in range(2):
            DX = res_list[i][index_func][index_init][cases_list[i]][0]
            DY = f_true_list[index_func](DX)

            f_pred_grids = sgp.fit(DX[:num_init + num_seq], 
                                   DY[:num_init + num_seq]).predict(grids)
            axes[k][1 + i].contourf(xv, yv, f_pred_grids.reshape(yv.shape), 
                                    levels=levels)
            axes[k][1 + i].scatter(DX[num_init: num_init + num_seq, 0], 
                                   DX[num_init: num_init + num_seq, 1], 
                                   s=4, color='tab:red')
            
            kde_c = custom_KDE(f_pred_grids.flatten(), weights=pdf_grids, bw=bw)
            x_min = min(kde_c.data.min(), kde_t.data.min()) 
            x_max = max(kde_c.data.max(), kde_t.data.max())
            rang = x_max-x_min
            x_eva = np.linspace(x_min - 0.01*rang,
                                x_max + 0.01*rang, 1024)
            yc = kde_c.evaluate(x_eva)
            axes[k][3].plot(x_eva, yc, color=color_list[i])
            
                
        for i in range(3):
                
            circle1, circle2 = (plt.Circle(max_pos, 0.5, color='k', 
                                            fill=False, linestyle='--'), 
                                plt.Circle(min_pos, 0.5, color='k', 
                                            fill=False, linestyle='--'))
            axes[k][i].add_patch(circle1)
            axes[k][i].add_patch(circle2)  
            axes[k][i].set_ylim(-5, 5)

        # Plot true PDF
        axes[k][3].set_yscale('log')
        axes[k][3].set_ylim([1e-12, 1e0])
        yt = kde_t.evaluate(x_eva)
        axes[k][3].plot(x_eva, yt, color='black')
        
        # Set ticks, labels and title
        if k == len(num_seq_list) - 1:
            for i in range(3):       
                axes[k][i].set_xlabel('$x_1$')
                axes[k][i].set_xticks([-5, 0, 5])
                axes[k][i].set_xlim(-5, 5)
            axes[k][3].set_xlabel('$f$')
            axes[k][3].set_xticks([-5, 0, 5])
        else:
            for i in range(3):
                axes[k][i].set_xticks([])
                axes[k][i].set_xlim(-5, 5)
            axes[k][3].set_xticks([])
        
        for i in range(3):
            if i == 0:
                axes[k][i].set_ylabel('$x_2$')
                axes[k][i].set_yticks([-5, 0, 5])
            else:
                axes[k][i].set_yticks([])
        
        # fourth column, add ticks and label at the right side
        axes[k][3].yaxis.tick_right()
        axes[k][3].yaxis.set_label_position("right")
        
        axes[k][3].text(-6, 1 * 1e-1, '$n_{seq}$=' + str(num_seq_list[k]), 
                        fontsize=14)

        
    axes[0][0].set_title('True function \n', fontsize=14)
    axes[0][1].set_title('Predicted function with \n sequential samples ($\\alpha=0)$', fontsize=14)
    axes[0][2].set_title('Predicted function with \n sequential samples ($\\alpha=3)$', fontsize=14)
    axes[0][3].set_title('Response PDF \n', fontsize=14)
        
        
    plt.subplots_adjust(wspace=0.2,
                        hspace=0.2)
    plt.tight_layout()
    if save_address:
        plt.savefig(save_address)
    plt.show()
    
    
def plot_3d(z, f, sgp, DX1, DX2, cir_position, save_address=None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    levels = np.arange(-8, 9)
    x_plot = np.linspace(-5,  5,  101)
    y_plot = np.linspace(-5,  5,  101)
    
    x_mesh_plot, y_mesh_plot = np.meshgrid(x_plot, y_plot)
    grids = np.concatenate((x_mesh_plot.reshape(-1, 1), 
                            y_mesh_plot.reshape(-1, 1), 
                            z * np.ones(y_mesh_plot.reshape(-1, 1).shape))
                           , axis=1)
    cs = axes[0].contourf(x_mesh_plot, y_mesh_plot, 
                 f(grids).reshape(x_mesh_plot.shape),
                 levels=levels)
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].set_yticks([-5, 0, 5])
    axes[0].set_xticks([-5, 0, 5])
    axes[0].set_title('True function')
    
    sgp.fit(DX1, f(DX1))
    axes[1].contourf(x_mesh_plot, y_mesh_plot, 
                 sgp.predict(grids).reshape(x_mesh_plot.shape), 
                 levels=cs.levels)
    axes[1].set_xlabel('$x_1$')
    axes[1].set_xlabel('$x_1$')
    axes[1].set_yticks([])
    axes[1].set_xticks([-5, 0, 5])
    axes[1].set_title('Predictive function $\\alpha=0$')
    
    sgp.fit(DX2, f(DX2))
    axes[2].contourf(x_mesh_plot, y_mesh_plot, 
                 sgp.predict(grids).reshape(x_mesh_plot.shape), 
                 levels=cs.levels)
    axes[2].set_xlabel('$x_1$')
    axes[2].set_yticks([])
    axes[2].set_xticks([-5, 0, 5])
    axes[2].set_title('Predictive function $\\alpha=3$')
    
    
    for i in range(3):
        circle1 = plt.Circle(cir_position, 0.5, color='red', fill=False, linestyle='--')
        axes[i].add_patch(circle1)

    
    plt.tight_layout()
    plt.colorbar(cs, ax=axes, anchor=(-0.2, 0.5))
    if save_address:
        plt.savefig(save_address)
    plt.show()