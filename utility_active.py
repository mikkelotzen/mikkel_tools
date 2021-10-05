import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as tick

# Color
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
projection = ccrs.Mollweide(central_longitude=0)
import matplotlib.colors as colors 

color_zesty_cbf = [(0.0,  0.10980392156862745,  0.30196078431372547), 
                   (0.5019607843137255,  0.6862745098039216,  1.0), 
                   (1, 1, 1), 
                   (1.0,  0.5372549019607843,  0.30196078431372547), 
                   (0.30196078431372547,  0.10196078431372549,  0.0)]  # dark bluish -> bright blue -> white -> bright orange -> darker orange

cm_zesty_cbf = LinearSegmentedColormap.from_list("zesty_cbf", color_zesty_cbf, N=10001)


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0.0, 0.5, 1.0]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_while_learning(epoch):
    # Plot learning curve
    fig = plt.figure(figsize=(9,9), constrained_layout=True) # Initiate figure with constrained layout
    gs = fig.add_gridspec(1, 2) # Add 1x2 grid
    ax1 = fig.add_subplot(gs[0, :]) 
    # Finalize plots   
    ax1.clear()
    ax1.set_title('Error CN')
    ax1.set_xlabel("Epoch")
    ax1.set_xlim([np.min(epoch_range),N_epochs])    
    #ax1.set_ylim([0.0,np.max([E_valid_collect,E_train_collect])])
    
    ax1.grid()
    ax1.semilogy(epoch_range, E_train_collect[:,0], '-', color="C0", label = "training")
    ax1.semilogy(epoch_range, E_valid_collect[:,0], '--', color="C1", label = "validation")
    ax1.text(0.5, 0.9, "Current lr: " + str(optimizer.param_groups[0]["lr"]),
            horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.legend(loc="upper right")
    
    fig.canvas.draw()

    fig.savefig('nets/training_sequences/training_test_{}'.format(epoch), bbox_inches='tight', dpi = 100)


    # Plot validation batch RMSE
    sat_p = sat_in_v[:,:].permute(1,0).detach().cpu().numpy()
    C_op = C_ov[:,:].permute(1,0).detach().cpu().numpy()
    Li_op = Li_ov[:,:].permute(1,0).detach().cpu().numpy()
    C_lp = C_lv[:,:].permute(1,0).detach().cpu().numpy()
    Li_lp = Li_lv[:,:].permute(1,0).detach().cpu().numpy()

    # Label
    clip_op = Li_op.copy()
    clip_op[:mt_util.shc_vec_len(n_cut_max)] += C_op[:mt_util.shc_vec_len(n_cut_max),:]
    
    clip_lp = Li_lp.copy()
    clip_lp[:mt_util.shc_vec_len(n_cut_max)] += C_lp[:mt_util.shc_vec_len(n_cut_max),:]
    
    sat_op = Gr@clip_op
    sat_lp = Gr@clip_lp

    rmse_v_b = np.sqrt(np.mean((sat_lp-sat_op)**2,axis=1))
    rmse_v = np.sqrt(np.mean((sat_lp-sat_op)**2,axis=0))

    fig = plt.figure(figsize=(9,9), constrained_layout=True) # Initiate figure with constrained layout
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.clear()
    ax1.set_title("Validation sat obs RMSE, mean over batch")
    ax1.set_xlabel("[nT]")
    ax1.set_ylabel("Count")    
    ax1.grid()
    ax1.hist(rmse_v_b.reshape(-1),bins=21)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.clear()
    ax2.set_title("Validation sat obs RMSE, mean over obs")
    ax2.set_xlabel("[nT]")
    ax2.set_ylabel("Count")    
    ax2.grid()
    ax2.hist(rmse_v.reshape(-1),bins=21)

    fig.canvas.draw()
    fig.savefig('nets/training_sequences/rmse_val_test_{}'.format(epoch), bbox_inches='tight', dpi = 100)


    # Plot training batch RMSE
    sat_p = sat_in_t[:,:].permute(1,0).detach().cpu().numpy()
    C_op = C_ot[:,:].permute(1,0).detach().cpu().numpy()
    Li_op = Li_ot[:,:].permute(1,0).detach().cpu().numpy()
    C_lp = C_lt[:,:].permute(1,0).detach().cpu().numpy()
    Li_lp = Li_lt[:,:].permute(1,0).detach().cpu().numpy()

    # Label
    clip_op = Li_op.copy()
    clip_op[:mt_util.shc_vec_len(n_cut_max)] += C_op[:mt_util.shc_vec_len(n_cut_max),:]
    
    clip_lp = Li_lp.copy()
    clip_lp[:mt_util.shc_vec_len(n_cut_max)] += C_lp[:mt_util.shc_vec_len(n_cut_max),:]
    
    sat_op = Gr@clip_op
    sat_lp = Gr@clip_lp

    rmse_v_b = np.sqrt(np.mean((sat_lp-sat_op)**2,axis=1))
    rmse_v = np.sqrt(np.mean((sat_lp-sat_op)**2,axis=0))

    fig = plt.figure(figsize=(9,9), constrained_layout=True) # Initiate figure with constrained layout
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.clear()
    ax1.set_title("Training sat obs RMSE, mean over batch")
    ax1.set_xlabel("[nT]")
    ax1.set_ylabel("Count")    
    ax1.grid()
    ax1.hist(rmse_v_b.reshape(-1),bins=21)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.clear()
    ax2.set_title("Training sat obs RMSE, mean over obs")
    ax2.set_xlabel("[nT]")
    ax2.set_ylabel("Count")    
    ax2.grid()
    ax2.hist(rmse_v.reshape(-1),bins=21)
    fig.canvas.draw()
    fig.savefig('nets/training_sequences/rmse_tra_test_{}'.format(epoch), bbox_inches='tight', dpi = 100)



    # Plot fit training
    fig = plt.figure(figsize=(9,6), constrained_layout=True) # Initiate figure with constrained layout
    gs = fig.add_gridspec(3, 3) # Add 3x3 grid
    ax1 = fig.add_subplot(gs[0, 0], projection=projection) 
    ax2 = fig.add_subplot(gs[0, 1], projection=projection)
    ax12 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0], projection=projection)
    ax4 = fig.add_subplot(gs[1, 1], projection=projection)
    ax34 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, 0], projection=projection)
    ax6 = fig.add_subplot(gs[2, 1], projection=projection)
    ax56 = fig.add_subplot(gs[2, 2])

    sat_p = sat_in_t[0,:].detach().cpu().numpy()
    C_op = C_ot[0,:].detach().cpu().numpy()
    Li_op = Li_ot[0,:].detach().cpu().numpy()
    C_lp = C_lt[0,:].detach().cpu().numpy()
    Li_lp = Li_lt[0,:].detach().cpu().numpy()
    
    # Input
    ax1.clear()
    ax1.set_title("Li+C input obs")
    im1 = ax1.scatter(clip.grid_phi, 90-clip.grid_theta, s=10, c=sat_p, marker = "o", 
                    transform=ccrs.PlateCarree(), rasterized=True, cmap=cm_zesty_cbf, 
                    norm = MidpointNormalize(midpoint=0.0)) #, vmin = -5*10**4, vmax = 5*10**4
    ax1.coastlines(linewidth = 0.2, color = (0.4,0.4,0.4))
    ax1.axis('off')
    
    # Label
    clip_op = Li_op.copy()
    #clip_op[:i_n_C] += C_op[:mt_util.shc_vec_len(20)]
    clip_op[:mt_util.shc_vec_len(n_cut_max)] += C_op[:mt_util.shc_vec_len(n_cut_max)]
    
    clip_lp = Li_lp.copy()
    #clip_lp[:i_n_C] += C_lp
    clip_lp[:mt_util.shc_vec_len(n_cut_max)] += C_lp[:mt_util.shc_vec_len(n_cut_max)]
    
    sat_op = Gr@clip_op
    sat_lp = Gr@clip_lp
    
    ax2.clear()
    ax2.set_title("Net output obs")
    im2 = ax2.scatter(clip.grid_phi, 90-clip.grid_theta, s=10, c=sat_op, marker = "o", 
                    transform=ccrs.PlateCarree(), rasterized=True, cmap=cm_zesty_cbf, 
                    norm = MidpointNormalize(midpoint=0.0)) #, vmin = -5*10**4, vmax = 5*10**4
    ax2.coastlines(linewidth = 0.2, color = (0.4,0.4,0.4))
    ax2.axis('off')
    
    ax12.clear()
    ax12.set_title("Residuals")
    ax12.hist((sat_op-sat_p).reshape(-1),bins=21)
    
    # C Label
    #C_lpm = Gr_C@C_lp
    C_lpm = Gr_C[:,:mt_util.shc_vec_len(n_cut_max)]@C_lp[:mt_util.shc_vec_len(n_cut_max)]
    ax3.clear()
    ax3.set_title("Dynamo simulation core")
    im3 = ax3.scatter(clip.grid_phi, 90-clip.grid_theta, s=10, c=C_lpm, marker = "o", 
                    transform=ccrs.PlateCarree(), rasterized=True, cmap=cm_zesty_cbf, 
                    norm = MidpointNormalize(midpoint=0.0)) #, vmin = -2*10**6, vmax = 2*10**6
    ax3.coastlines(linewidth = 0.2, color = (0.4,0.4,0.4))
    ax3.axis('off')
    
    # C output
    C_opm = Gr_C[:,:mt_util.shc_vec_len(n_cut_max)]@C_op[:mt_util.shc_vec_len(n_cut_max)]
    #C_opm = Gr_C@C_op
    ax4.clear()
    ax4.set_title("Net output shc core")
    im4 = ax4.scatter(clip.grid_phi, 90-clip.grid_theta, s=10, c=C_opm, marker = "o", 
                    transform=ccrs.PlateCarree(), rasterized=True, cmap=cm_zesty_cbf, 
                    norm = MidpointNormalize(midpoint=0.0)) #, vmin = -2*10**6, vmax = 2*10**6
    ax4.coastlines(linewidth = 0.2, color = (0.4,0.4,0.4))
    ax4.axis('off')
    
    ax34.clear()
    ax34.set_title("Residuals")
    ax34.hist((C_opm-C_lpm).reshape(-1),bins=21)
    
    # Li Label
    Li_lpm = Gr_Li@Li_lp
    ax5.clear()
    ax5.set_title("Crustal lith")
    im5 = ax5.scatter(clip.grid_phi, 90-clip.grid_theta, s=10, c=Li_lpm, marker = "o", 
                    transform=ccrs.PlateCarree(), rasterized=True, cmap=cm_zesty_cbf, 
                    norm = MidpointNormalize(midpoint=0.0)) #, vmin = -3*10**2, vmax = 3*10**2
    ax5.coastlines(linewidth = 0.2, color = (0.4,0.4,0.4))
    ax5.axis('off')
    
    # Li output
    Li_opm = Gr_Li@Li_op
    ax6.clear()
    ax6.set_title("Net output shc lith")
    im6 = ax6.scatter(clip.grid_phi, 90-clip.grid_theta, s=10, c=Li_opm, marker = "o", 
                    transform=ccrs.PlateCarree(), rasterized=True, cmap=cm_zesty_cbf, 
                    norm = MidpointNormalize(midpoint=0.0)) #, vmin = -3*10**2, vmax = 3*10**2
    ax6.coastlines(linewidth = 0.2, color = (0.4,0.4,0.4)) 
    ax6.axis('off')

    ax56.clear()
    ax56.set_title("Residuals")
    ax56.hist((Li_opm-Li_lpm).reshape(-1),bins=21)
    
    # End
    fig.canvas.draw()
    fig.savefig('nets/training_sequences/fit_test_tra_{}'.format(epoch), bbox_inches='tight', dpi = 100)


    # Plot fit
    fig = plt.figure(figsize=(9,6), constrained_layout=True) # Initiate figure with constrained layout
    gs = fig.add_gridspec(3, 3) # Add 3x3 grid
    ax1 = fig.add_subplot(gs[0, 0], projection=projection) 
    ax2 = fig.add_subplot(gs[0, 1], projection=projection)
    ax12 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0], projection=projection)
    ax4 = fig.add_subplot(gs[1, 1], projection=projection)
    ax34 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, 0], projection=projection)
    ax6 = fig.add_subplot(gs[2, 1], projection=projection)
    ax56 = fig.add_subplot(gs[2, 2])

    sat_p = sat_in_v[0,:].detach().cpu().numpy()
    C_op = C_ov[0,:].detach().cpu().numpy()
    Li_op = Li_ov[0,:].detach().cpu().numpy()
    C_lp = C_lv[0,:].detach().cpu().numpy()
    Li_lp = Li_lv[0,:].detach().cpu().numpy()
    
    # Input
    ax1.clear()
    ax1.set_title("Li+C input obs")
    im1 = ax1.scatter(clip.grid_phi, 90-clip.grid_theta, s=10, c=sat_p, marker = "o", 
                    transform=ccrs.PlateCarree(), rasterized=True, cmap=cm_zesty_cbf, 
                    norm = MidpointNormalize(midpoint=0.0)) #, vmin = -5*10**4, vmax = 5*10**4
    ax1.coastlines(linewidth = 0.2, color = (0.4,0.4,0.4))
    ax1.axis('off')
    
    # Label
    clip_op = Li_op.copy()
    #clip_op[:i_n_C] += C_op[:mt_util.shc_vec_len(20)]
    clip_op[:mt_util.shc_vec_len(n_cut_max)] += C_op[:mt_util.shc_vec_len(n_cut_max)]
    
    clip_lp = Li_lp.copy()
    #clip_lp[:i_n_C] += C_lp
    clip_lp[:mt_util.shc_vec_len(n_cut_max)] += C_lp[:mt_util.shc_vec_len(n_cut_max)]
    
    sat_op = Gr@clip_op
    sat_lp = Gr@clip_lp
    
    ax2.clear()
    ax2.set_title("Net output obs")
    im2 = ax2.scatter(clip.grid_phi, 90-clip.grid_theta, s=10, c=sat_op, marker = "o", 
                    transform=ccrs.PlateCarree(), rasterized=True, cmap=cm_zesty_cbf, 
                    norm = MidpointNormalize(midpoint=0.0)) #, vmin = -5*10**4, vmax = 5*10**4
    ax2.coastlines(linewidth = 0.2, color = (0.4,0.4,0.4))
    ax2.axis('off')
    
    ax12.clear()
    ax12.set_title("Residuals")
    ax12.hist((sat_op-sat_p).reshape(-1),bins=21)
    
    # C Label
    #C_lpm = Gr_C@C_lp
    C_lpm = Gr_C[:,:mt_util.shc_vec_len(n_cut_max)]@C_lp[:mt_util.shc_vec_len(n_cut_max)]
    ax3.clear()
    ax3.set_title("Dynamo simulation core")
    im3 = ax3.scatter(clip.grid_phi, 90-clip.grid_theta, s=10, c=C_lpm, marker = "o", 
                    transform=ccrs.PlateCarree(), rasterized=True, cmap=cm_zesty_cbf, 
                    norm = MidpointNormalize(midpoint=0.0)) #, vmin = -2*10**6, vmax = 2*10**6
    ax3.coastlines(linewidth = 0.2, color = (0.4,0.4,0.4))
    ax3.axis('off')
    
    # C output
    C_opm = Gr_C[:,:mt_util.shc_vec_len(n_cut_max)]@C_op[:mt_util.shc_vec_len(n_cut_max)]
    #C_opm = Gr_C@C_op
    ax4.clear()
    ax4.set_title("Net output shc core")
    im4 = ax4.scatter(clip.grid_phi, 90-clip.grid_theta, s=10, c=C_opm, marker = "o", 
                    transform=ccrs.PlateCarree(), rasterized=True, cmap=cm_zesty_cbf, 
                    norm = MidpointNormalize(midpoint=0.0)) #, vmin = -2*10**6, vmax = 2*10**6
    ax4.coastlines(linewidth = 0.2, color = (0.4,0.4,0.4))
    ax4.axis('off')
    
    ax34.clear()
    ax34.set_title("Residuals")
    ax34.hist((C_opm-C_lpm).reshape(-1),bins=21)
    
    # Li Label
    Li_lpm = Gr_Li@Li_lp
    ax5.clear()
    ax5.set_title("Crustal lith")
    im5 = ax5.scatter(clip.grid_phi, 90-clip.grid_theta, s=10, c=Li_lpm, marker = "o", 
                    transform=ccrs.PlateCarree(), rasterized=True, cmap=cm_zesty_cbf, 
                    norm = MidpointNormalize(midpoint=0.0)) #, vmin = -3*10**2, vmax = 3*10**2
    ax5.coastlines(linewidth = 0.2, color = (0.4,0.4,0.4))
    ax5.axis('off')
    
    # Li output
    Li_opm = Gr_Li@Li_op
    ax6.clear()
    ax6.set_title("Net output shc lith")
    im6 = ax6.scatter(clip.grid_phi, 90-clip.grid_theta, s=10, c=Li_opm, marker = "o", 
                    transform=ccrs.PlateCarree(), rasterized=True, cmap=cm_zesty_cbf, 
                    norm = MidpointNormalize(midpoint=0.0)) #, vmin = -3*10**2, vmax = 3*10**2
    ax6.coastlines(linewidth = 0.2, color = (0.4,0.4,0.4)) 
    ax6.axis('off')

    ax56.clear()
    ax56.set_title("Residuals")
    ax56.hist((Li_opm-Li_lpm).reshape(-1),bins=21)
    
    # End
    fig.canvas.draw()
    fig.savefig('nets/training_sequences/fit_test_val_{}'.format(epoch), bbox_inches='tight', dpi = 100)


    # P spec
    C_op = C_ot[:5,:].detach().cpu().numpy()
    Li_op = Li_ot[:5,:].detach().cpu().numpy()
    C_lp = C_lt[:5,:].detach().cpu().numpy()
    Li_lp = Li_lt[:5,:].detach().cpu().numpy()

    nmax_pairs = np.ones(5,dtype=int)*int(n_max_C)
    label = ["1","2","3","4","5"]
    mt_util.plot_p_spec(C_op, clip.r_cmb, n_max_C, g_spec_compares = C_lp, nmax_pairs = nmax_pairs, 
                        nmax_pairs_compare = nmax_pairs, spec_style="pair_compare", figsize=(9,9), label=label,
                        savefig = True, save_string = 'C_test_tra_{}'.format(epoch),
                        save_folder="nets/training_sequences/")


    nmax_pairs = np.ones(5,dtype=int)*int(n_max_Li)
    label = ["1","2","3","4","5"]
    mt_util.plot_p_spec(Li_op, clip.a, n_max_Li, g_spec_compares = Li_lp, nmax_pairs = nmax_pairs, 
                        nmax_pairs_compare = nmax_pairs, spec_style="pair_compare", figsize=(9,9), label=label,
                        savefig = True, save_string = 'Li_test_tra_{}'.format(epoch),
                        save_folder="nets/training_sequences/")


    # P spec
    C_op = C_ov[:5,:].detach().cpu().numpy()
    Li_op = Li_ov[:5,:].detach().cpu().numpy()
    C_lp = C_lv[:5,:].detach().cpu().numpy()
    Li_lp = Li_lv[:5,:].detach().cpu().numpy()

    nmax_pairs = np.ones(5,dtype=int)*int(n_max_C)
    label = ["1","2","3","4","5"]
    mt_util.plot_p_spec(C_op, clip.r_cmb, n_max_C, g_spec_compares = C_lp, nmax_pairs = nmax_pairs, 
                        nmax_pairs_compare = nmax_pairs, spec_style="pair_compare", figsize=(9,9), label=label,
                        savefig = True, save_string = 'C_test_val_{}'.format(epoch),
                        save_folder="nets/training_sequences/")


    nmax_pairs = np.ones(5,dtype=int)*int(n_max_Li)
    label = ["1","2","3","4","5"]
    mt_util.plot_p_spec(Li_op, clip.a, n_max_Li, g_spec_compares = Li_lp, nmax_pairs = nmax_pairs, 
                        nmax_pairs_compare = nmax_pairs, spec_style="pair_compare", figsize=(9,9), label=label,
                        savefig = True, save_string = 'Li_test_val_{}'.format(epoch),
                        save_folder="nets/training_sequences/")