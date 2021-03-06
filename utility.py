"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Utility functions by Mikkel Otzen
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os
import cartopy.crs as ccrs
import time
import numpy as np

utility_abs_path = os.path.dirname(__file__)

color_zesty_cbf = [(0.0,  0.10980392156862745,  0.30196078431372547), 
                   (0.5019607843137255,  0.6862745098039216,  1.0), 
                   (1, 1, 1), 
                   (1.0,  0.5372549019607843,  0.30196078431372547), 
                   (0.30196078431372547,  0.10196078431372549,  0.0)]  # dark bluish -> bright blue -> white -> bright orange -> darker orange

cm_zesty_cbf = LinearSegmentedColormap.from_list("zesty_cbf", color_zesty_cbf, N=10001)

color_rgb_zesty_pos = (1.0,  0.5372549019607843,  0.30196078431372547)
color_rgb_zesty_neg = (0.5019607843137255,  0.6862745098039216,  1.0)

def dict_save(path, name, variable ):
    import pickle
    with open('%s' %path + name + '.pkl', 'wb') as f:
        pickle.dump(variable, f, pickle.HIGHEST_PROTOCOL)

def dict_load(path, name ):
    import pickle
    with open('%s' %path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def variable_save(filename, variable):
    import numpy as np
    
    np.save(filename,variable) # Save variable
    print('Saved file:', filename) 

def variable_load(filename):
    import numpy as np
    
    variable = np.load(filename) # Load variable
    
    print('Finished loading:', filename)
    return np.array(variable) 

def load_shc(filepath, comments = "#"):
    import numpy as np

    file = open(filepath)
    lines = file.readlines()
    n_comments = 0
    for each_line in lines:
        if each_line[0] == comments:
            n_comments += 1
        else:
            break

    file = np.loadtxt(filepath, comments=comments, skiprows=n_comments+2)
    return file

# Function for testing positive definiteness
def is_pos_def(A):
    if np.allclose(A, A.conj().T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def printProgressBar (iteration, total, *args, subject='', prefix = '', suffix = '', decimals = 1, length = 10, fill = 'O', notebook_style = True, sleep_time = 0.0001):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    
    if args:
        if notebook_style == True:
            print("\r%s |%s| %s%% %s %s. Counter: %s/%s, Running error magnitude: %.1f" % (prefix, bar, percent, suffix, subject, iteration, total, args[0]), end = "")
        else:
            print('\r%s |%s| %s%% %s %s. Counter: %s/%s, Running error magnitude: %.1f' % (prefix, bar, percent, suffix, subject, iteration, total, args[0]), end = '\r')
    else:
        if notebook_style == True:
            print("\r%s |%s| %s%% %s %s. Counter: %s/%s" % (prefix, bar, percent, suffix, subject, iteration, total), end = "")
        else:
            print('\r%s |%s| %s%% %s %s. Counter: %s/%s' % (prefix, bar, percent, suffix, subject, iteration, total), end = '\r')
    #time.sleep(sleep_time)
    # Print New Line on Complete
    if iteration == total: 
        print()


def plot_cartopy_global(lat = None, lon = None, data=None, limits_data = None, shape = (360,720), plot_quality = None,
                        unit = "[nT]", cmap = plt.cm.RdBu_r, projection_transformation = "Mollweide", figsize=(10,10),
                        title='Cartopy Earth plot', lat_0 = 0.0, lon_0 = 0.0, point_size=2, showfig=True, norm_class = False,
                        scale_uneven = False, flip_shape = False, flip_grid = True, transpose_grid = False, shift_grid = False,
                        savefig = False, dpi = 100, path = None, saveformat = ".png"):

    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    import numpy as np
    import matplotlib.colors as colors
    import matplotlib.colorbar
    
    # Start figure
    fig = plt.figure(figsize=figsize)
    
    vmin = np.min(limits_data)
    vmax = np.max(limits_data)
    
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0.0, 0.5, 1.0]
            return np.ma.masked_array(np.interp(value, x, y))

    class SqueezedNorm(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, mid=0, s1=1.75, s2=1.75, clip=False):
            self.vmin = vmin # minimum value
            self.mid  = mid  # middle value
            self.vmax = vmax # maximum value
            self.s1=s1; self.s2=s2
            f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
            self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                                 f(x,zero,vmin,s2)*(x<zero)+0.5
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
            return np.ma.masked_array(r)
    
    # Plotting ranges and norms
    if limits_data is None:
        vmin = np.min(data)
        vmax = np.max(data)

    if scale_uneven == False:
        veven = np.max([abs(vmax),abs(vmin)])
        vmin = -veven
        vmax = veven
        norm_in = None
    else:
        scale_diff = vmax-vmin
        vmin = (vmax - scale_diff)

        if norm_class == "midpoint":
            norm_in = MidpointNormalize(midpoint=0.)
        elif norm_class == "squeezed":
            norm_in = SqueezedNorm()
        else:
            norm_in = None
    
    # Plotting init
    if projection_transformation == "ortho":
        projection = ccrs.Orthographic(central_longitude=lon_0, central_latitude=lat_0)
    else:
        projection = ccrs.Mollweide()
    
    
    if plot_quality == "high":
        axes_class = (GeoAxes, dict(map_projection=projection))
        # axgr = AxesGrid(fig, 111, axes_class=axes_class,
        #                 nrows_ncols=(1, 1),
        #                 axes_pad=0.1,
        #                 cbar_location='bottom',
        #                 cbar_mode='single',
        #                 cbar_pad=0.05,
        #                 cbar_size='5%',
        #                 label_mode='')  # note the empty label_mode

        axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(1, 1),
                label_mode='')  # note the empty label_mode

        axgr[0].coastlines()
        axgr[0].set_global()
        
        if data is None:
            axgr[0].scatter(lon, lat, s=point_size, transform=ccrs.PlateCarree(), cmap=cm_zesty_cbf)

        else:
            cb = axgr[0].scatter(lon, lat, s=point_size, c=data, transform=ccrs.PlateCarree(), vmin = vmin, vmax = vmax, cmap=cm_zesty_cbf, norm = norm_in)

            cax,kw = matplotlib.colorbar.make_axes(axgr[0],location='bottom', shrink=0.7, aspect=40, fraction = 0.23)
            out=fig.colorbar(cb,cax=cax,extend='neither',**kw)
            out.set_label('%s %s' %(title,unit), size=10)

            #axgr.cbar_axes[0].colorbar(cb)
            #cax = axgr.cbar_axes[0]
            #axis = cax.axis[cax.orientation]
            #axis.label.set_text('%s %s' %(title,unit))
    else:

        if flip_shape == True:
            shape = (shape[1], shape[0])

        ax = plt.axes(projection=projection)

        ax.coastlines()
        ax.set_global()

        data_in = np.ravel(data).reshape(shape[0],shape[1])

        if transpose_grid == True:
            data_in = data_in.T

        if flip_grid == True:
            data_in = np.flipud(data_in)

        if shift_grid == True:
            shift_axis = np.min(shape)
            data_in = np.hstack((data_in[:,shift_axis:],data_in[:,:shift_axis]))

        cs = ax.imshow(data_in,  vmin = vmin, vmax = vmax, cmap = cm_zesty_cbf, norm=norm_in, transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])

        cax,kw = matplotlib.colorbar.make_axes(ax,location='bottom', pad=0.02, shrink=0.7, aspect=60)
        out=fig.colorbar(cs,cax=cax,extend='neither',**kw)
        out.set_label('%s %s' %(title,unit), size=10)
        ax.background_patch.set_fill(False)
        
    if savefig is True and path is not None:
        title_to_filename = title.replace(" ", "_").replace(":","").replace("-","_").replace("/","").replace("(","").replace(")","")
        plt.savefig('%s%s%s' %(path,title_to_filename,saveformat), bbox_inches='tight', dpi = dpi, format="%s" %saveformat.replace(".",""))
    if showfig is True:
        plt.show()
    return

def plot_cartopy_animation(lat = None, lon = None, data=None, limits_data = None, shape = (360,720), animation_quality = None, frames = 2, interval = 200, projection_transformation = "Mollweide", unit = "[nT]", title = "Cartopy Earth Plot", cmap = plt.cm.RdBu_r, figsize=(10,10), point_size=1, norm_class = False, scale_uneven = False, flip_shape = False, flip_grid = True, transpose_grid = False, shift_grid = False, animation_output = "javascript", filename = "", path_save_mp4 = "images/"):

    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    import numpy as np
    import matplotlib.colors as colors
    import matplotlib.colorbar
    
    # animation
    from matplotlib import animation, rc
    from IPython.display import HTML, Image, display, Video
    import os
    
    if data is None:
        raise ValueError("No data accessible for animation")
    
    if animation_output == "html5":
        html = "html5"
    else:
        html = "jshtml"
        
    rc('animation', html=html)
        
    
    # Start figure
    fig = plt.figure(figsize=figsize)
    
    vmin = np.min(limits_data)
    vmax = np.max(limits_data)
    
    # COLORBAR TRANSFORM CLASSES
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            # I'm ignoring masked values and all kinds of edge cases to make a
            # simple example...
            x, y = [self.vmin, self.midpoint, self.vmax], [0.0, 0.5, 1.0]
            return np.ma.masked_array(np.interp(value, x, y))

    class SqueezedNorm(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, mid=0, s1=1.75, s2=1.75, clip=False):
            self.vmin = vmin # minimum value
            self.mid  = mid  # middle value
            self.vmax = vmax # maximum value
            self.s1=s1; self.s2=s2
            f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
            self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                                 f(x,zero,vmin,s2)*(x<zero)+0.5
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
            return np.ma.masked_array(r)
    
    # Plotting ranges and norms
    if limits_data is None:
        vmin = np.min(data)
        vmax = np.max(data)

    if scale_uneven == False:
        veven = np.max([abs(vmax),abs(vmin)])
        vmin = -veven
        vmax = veven
        norm_in = None

    else:
        scale_diff = vmax-vmin
        vmin = (vmax - scale_diff)

        if norm_class == "midpoint":
            norm_in = MidpointNormalize(midpoint=0.)
        elif norm_class == "squeezed":
            norm_in = SqueezedNorm()
        else:
            norm_in = None
    
    # Plotting init
    if projection_transformation == "ortho":
        projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=0.0)
    else:
        projection = ccrs.Mollweide()
    

    if animation_quality == "high":
        axes_class = (GeoAxes, dict(map_projection=projection))
        # axgr = AxesGrid(fig, 111, axes_class=axes_class,
        #                 nrows_ncols=(1, 1),
        #                 axes_pad=0.1,
        #                 cbar_location='bottom',
        #                 cbar_mode='single',
        #                 cbar_pad=0.05,
        #                 cbar_size='5%',
        #                 label_mode='')  # note the empty label_mode

        axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(1, 1),
                label_mode='')  # note the empty label_mode

        axgr[0].coastlines()
        axgr[0].set_global()
        

        cb = axgr[0].scatter(lon, lat, s=point_size, c=limits_data, transform=ccrs.PlateCarree(), vmin = vmin, vmax = vmax, cmap=cm_zesty_cbf, norm = norm_in)

        cax,kw = matplotlib.colorbar.make_axes(axgr[0],location='bottom', shrink=0.7, aspect=40, fraction = 0.23)
        out=fig.colorbar(cb,cax=cax,extend='neither',**kw)
        out.set_label('%s %s' %(title,unit), size=10)
    
    #if animation_quality == "high":
    #    axes_class = (GeoAxes, dict(map_projection=projection))
    #    axgr = AxesGrid(fig, 111, axes_class=axes_class,
    #                    nrows_ncols=(1, 1),
    #                    axes_pad=0.1,
    #                    cbar_location='bottom',
    #                    cbar_mode='single',
    #                    cbar_pad=0.05,
    #                    cbar_size='5%',
    #                    label_mode='')  # note the empty label_mode

        # axgr[0].coastlines()
        # axgr[0].set_global()
                
        # cb = axgr[0].scatter(lon, lat, s=point_size, c=limits_data, transform=ccrs.PlateCarree(), vmin = vmin, vmax = vmax, cmap=cmap, norm = norm_in)

        # axgr.cbar_axes[0].colorbar(cb)
        # cax = axgr.cbar_axes[0]
        # axis = cax.axis[cax.orientation]
        # axis.label.set_text('%s %s' %(title,unit))

        # cb = axgr[0].scatter(lon, lat, s=point_size, c=data, transform=ccrs.PlateCarree(), vmin = vmin, vmax = vmax, cmap=cmap, norm = norm_in)

        # cax,kw = matplotlib.colorbar.make_axes(axgr[0],location='bottom', shrink=0.7, aspect=40, fraction = 0.23)
        # out=fig.colorbar(cb,cax=cax,extend='neither',**kw)
        # out.set_label('%s %s' %(title,unit), size=10)

        def animate(i):
            cb = axgr[0].scatter(lon, lat, s=point_size, c=data[:,i], transform=ccrs.PlateCarree(), cmap=cm_zesty_cbf, norm = norm_in, vmin = vmin, vmax = vmax)
            return (cb,)
        
    else:

        ax = plt.axes(projection=projection)

        ax.coastlines()
        ax.set_global()

        if flip_shape == True:
            shape = (shape[1], shape[0])

        data_init = np.ravel(limits_data).reshape(shape[0],shape[1])

        if transpose_grid == True:
            data_init = data_init.T

        if flip_grid == True:
            data_init = np.flipud(data_init)

        #data_init = np.flipud(np.ravel(limits_data).reshape(shape[0],shape[1]))
        if shift_grid == True:
            data_init = np.hstack((data_init[:,shape[0]:],data_init[:,:shape[0]]))

        cs = ax.imshow(data_init,  vmin = vmin, vmax = vmax, cmap = cm_zesty_cbf, norm=norm_in, transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])

        cax,kw = matplotlib.colorbar.make_axes(ax,location='bottom', pad=0.02, shrink=0.7, aspect=60)
        out=fig.colorbar(cs,cax=cax,extend='neither',**kw)
        out.set_label('%s %s' %(title,unit), size=10)
        
        def animate(i):
            data_i = data[:,i]

            data_i = np.ravel(data_i).reshape(shape[0],shape[1])

            if transpose_grid == True:
                data_i = data_i.T

            if flip_grid == True:
                data_i = np.flipud(data_i)                

            #data_i = np.flipud(np.ravel(data_i).reshape(shape[0],shape[1]))
            if shift_grid == True:
                data_i = np.hstack((data_i[:,shape[0]:],data_i[:,:shape[0]]))
            cs = ax.imshow(data_i,  vmin = vmin, vmax = vmax, cmap = cm_zesty_cbf, norm=norm_in, transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])
            return (cs,)

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval)
    
    plt.close() # Close the active figure to avoid extra plot
    
  
    if animation_output == "html5":
        if os.path.exists('rm ./{}{}.mp4'.format(path_save_mp4,filename)):
            os.remove('rm ./{}{}.mp4'.format(path_save_mp4,filename))
            
        fps = int(frames/(frames*(interval/1000)))
        anim.save('{}{}.mp4'.format(path_save_mp4,filename), fps = fps, writer='ffmpeg')
        return HTML('<left><video controls autoplay loop src="{}{}.mp4?{}" width=100%/></left>'.format(path_save_mp4,filename,int(np.random.uniform(1,10e20))))
    
    else:
        return anim
        #return HTML(anim.to_jshtml())
    #return HTML(anim.to_html5_video())
    #return anim

def plot_sdssim_reproduce(seqsim_obj, seqsim_res, m_equiv_lsq = None, m_mode = None, pos_mean_res = None, truth_obj = None, uncon_obj = None, lwidth = 1, lwidth_mult = 2, lwidth_div = 5,
                          lags_use = 300, hist_bins = 100, hist_pos_mean = True, hist_ti_ens = False, hist_density = False, hist_local_posterior = False, hist_ti_ens_limit = None,
                          res_bins = 200, spec_use = True, spec_step = 5, spec_r_at = None, spec_r_ref = 6371.2, spec_show_differences = True, spec_mag = True, sv_pos_mean = True,
                          sv_use = True, spec_ti_ens = False, res_use = True, unit_transform_n_to_m = False, patch_legend = False, ens_prior = False,
                          model_dict = None, spec_chaos_time = [2020,1,1], unit_var = "[nT²]", unit_lag = "[km]", unit_field = "[nT]", unit_res = "[nT]",
                          left=0.08, bottom=0.12, right=0.92, top=0.95, wspace = 0.2, hspace=0.25, label_fontsize = "small", res_power_format = False, res_print_f = 2, power_limit = 6,
                          tile_size_row = 3, tile_size_column = 2, figsize=(9,14), savefig = False, save_string = "", save_ftype = "pdf", save_dpi = 300, save_path = ""):
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy as sp
    import scipy.io as spio
    import pyshtools
    import matplotlib.ticker as tick

    # ChaosMagPy modules
    from chaosmagpy import load_CHAOS_matfile
    from chaosmagpy.model_utils import synth_values
    from chaosmagpy.data_utils import mjd2000

    # Scale from nano to mili
    if unit_transform_n_to_m == True:
            posterior_fields = seqsim_obj.m_DSS*10**(-6)
            if ens_prior == True:
                prior_ens = seqsim_obj.m_ens*10**(-6)
            #seqsim_res = seqsim_res*10**(-6)
            seqsim_data = seqsim_obj.data*10**(-6)
            if m_equiv_lsq is not None:
                m_equiv_lsq = m_equiv_lsq*10**(-6)
                lsq_equiv_res = seqsim_obj.lsq_equiv_res
            if m_mode is not None:
                m_mode = m_mode*10**(-6)
                mode_res = seqsim_obj.mode_res
            if truth_obj is not None:
                truth_data = truth_obj.data*10**(-6)
            if uncon_obj is not None:
                uncon_posterior = uncon_obj.m_DSS*10**(-6)
                uncon_data = uncon_obj.data*10**(-6)
    else:
        posterior_fields = seqsim_obj.m_DSS
        if ens_prior == True:
            prior_ens = seqsim_obj.m_ens
        seqsim_data = seqsim_obj.data
        if m_equiv_lsq is not None:
            m_equiv_lsq = m_equiv_lsq
            lsq_equiv_res = seqsim_obj.lsq_equiv_res
        if m_mode is not None:
            m_mode = m_mode
            mode_res = seqsim_obj.mode_res
        if truth_obj is not None:
            truth_data = truth_obj.data
        if uncon_obj is not None:
            uncon_posterior = uncon_obj.m_DSS
            uncon_data = uncon_obj.data

    SF = tick.ScalarFormatter() # Formatter for colorbar
    SF.set_powerlimits((power_limit, power_limit)) # Set sci exponent used

    N_sim = seqsim_res.shape[1]

    rmse_leg = np.sqrt(np.mean(np.power(seqsim_res,2),axis=0))

    color_rgb = (0.6,0.6,0.6)

    fig = plt.figure(figsize=figsize, constrained_layout=False, dpi = save_dpi) # Initiate figure with constrained layout

    if spec_use == True:
        # Generate ratio lists
        h_ratio = [1]*tile_size_row
        w_ratio = [1]*tile_size_column
        gs = fig.add_gridspec(tile_size_row, tile_size_column, height_ratios=h_ratio, width_ratios=w_ratio) # Add x-by-y grid
    elif sv_use == False:
        # Generate ratio lists
        h_ratio = [1]*(tile_size_row-1)
        w_ratio = [1]*tile_size_column
        gs = fig.add_gridspec(tile_size_row-1, tile_size_column, height_ratios=h_ratio, width_ratios=w_ratio) # Add x-by-y grid
    else:
        # Generate ratio lists
        h_ratio = [1]*(tile_size_row-1)
        w_ratio = [1]*tile_size_column
        gs = fig.add_gridspec(tile_size_row-1, tile_size_column, height_ratios=h_ratio, width_ratios=w_ratio) # Add x-by-y grid

    
    #### RESIDUALS ####
    if res_use == True:
        ax = fig.add_subplot(gs[0, 0])

        if res_power_format == True:
            ax.xaxis.set_major_formatter(SF)


        for i in np.arange(0,N_sim):
            y,binEdges=np.histogram(seqsim_res[:,[i]],bins=res_bins)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            if i == 0:
                ax.plot(bincenters, y, '-', color = color_rgb_zesty_pos, label='Posterior', linewidth = lwidth) 
            else:
                ax.plot(bincenters, y, '-', color = color_rgb_zesty_pos, linewidth = lwidth) 
        leg1 = mpatches.Patch(color=color_rgb_zesty_pos, label='Posterior')
        leg_handle = [leg1]

        if pos_mean_res is not None:
            y,binEdges=np.histogram(pos_mean_res,bins=res_bins)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            ax.plot(bincenters, y, '-', color = color_rgb_zesty_neg, label='Posterior mean', linewidth = lwidth_mult*lwidth)
            legpm = mpatches.Patch(color=color_rgb_zesty_neg, label='Posterior mean')
            leg_handle.append(legpm)

        if m_equiv_lsq is not None:
            y,binEdges=np.histogram(lsq_equiv_res,bins=res_bins)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            ax.plot(bincenters, y, '-', color = "C3", label='Equivalent LSQ', linewidth = lwidth_mult*lwidth)
            leg2 = mpatches.Patch(color="C3", label='Equivalent LSQ')
            leg_handle.append(leg2)

        if m_mode is not None:
            y,binEdges=np.histogram(mode_res,bins=res_bins)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            ax.plot(bincenters, y, '-', color = "C4", label='Maximum of marginal posterior', linewidth = lwidth_mult*lwidth) #,linestyle = "dashed"
            legmode = mpatches.Patch(color="C4", label='Maximum of marginal posterior')
            leg_handle.append(legmode)

        ax.set_title('(a) Observation estimate residuals')
        ax.annotate("Mean RMSE: {:.{}f}".format(np.mean(rmse_leg), res_print_f), (0.05, 0.5), xycoords='axes fraction', va='center', fontsize = label_fontsize)
        ax.set_xlabel("Field residuals {}".format(unit_res))
        ax.set_ylabel("Count")

        if patch_legend == True:
            ax.legend(handles=leg_handle, numpoints=1, labelspacing=1, loc='best', fontsize=label_fontsize, frameon=False)
        else:
            ax.legend(loc='best', fontsize = label_fontsize)

    #### HISTOGRAM ####
    if res_use == False:
        ax = fig.add_subplot(gs[0, :])
        plot_letter = "a"
    else:
        plot_letter = "b"
        ax = fig.add_subplot(gs[0, 1])

    # Unconditional posterior / Prior
    if uncon_obj is not None:
        for i in np.arange(0,uncon_obj.N_sim):
            y,binEdges=np.histogram(uncon_posterior[:,[i]], bins=hist_bins, density = hist_density)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            if i == 0:
                ax.plot(bincenters,y,'-', color = color_rgb, label='Prior', linewidth = lwidth/lwidth_div, zorder=0)  
            else:
                ax.plot(bincenters,y,'-', color = color_rgb, linewidth = lwidth/lwidth_div, zorder=0)     
        leg0 = mpatches.Patch(color=color_rgb, label='Prior')

    # Posterior
    for i in np.arange(0,N_sim):
        y,binEdges=np.histogram(posterior_fields[:,[i]], bins=hist_bins, density = hist_density)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        if i == 0:
            ax.plot(bincenters,y,'-', color = color_rgb_zesty_pos, label='Posterior', linewidth = lwidth)  
        else:
            ax.plot(bincenters,y,'-', color = color_rgb_zesty_pos, linewidth = lwidth)     
    leg1 = mpatches.Patch(color=color_rgb_zesty_pos, label='Posterior')

    # Local posterior
    if hist_local_posterior == True:
        for i in np.arange(0,len(posterior_fields)):
            y,binEdges=np.histogram(posterior_fields[[i],:], bins=hist_bins, density = hist_density)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            if i == 0:
                ax.plot(bincenters,y,'-', color = color_rgb_zesty_neg, label='Local posterior', linewidth = lwidth)  
            else:
                ax.plot(bincenters,y,'-', color = color_rgb_zesty_neg, linewidth = lwidth)   
        leg2 = mpatches.Patch(color=color_rgb_zesty_neg, label='Local posterior')

    # Posterior mean
    if hist_pos_mean == True:
        y,binEdges=np.histogram(np.mean(posterior_fields,axis=1), bins=hist_bins, density = hist_density)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        ax.plot(bincenters, y, '-', color = color_rgb_zesty_neg, label='Posterior mean', linewidth = lwidth_mult*lwidth)  
        leg2 = mpatches.Patch(color=color_rgb_zesty_neg, label='Posterior mean')

    # Equivalent lsq
    if m_equiv_lsq is not None:
        y,binEdges=np.histogram(np.array(m_equiv_lsq), bins=hist_bins, density = hist_density)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        ax.plot(bincenters, y, '--', color = 'C3', label='Equivalent LSQ', linestyle = "dashed", linewidth = lwidth_mult*lwidth) 
        leglsq = mpatches.Patch(color="C3", label='Equivalent LSQ')

    # Posterior mode
    if m_mode is not None:
        y,binEdges=np.histogram(m_mode, bins=hist_bins, density = hist_density)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        ax.plot(bincenters, y, '-', color = 'C4', label='Maximum of marginal posterior', linewidth = lwidth_mult*lwidth) 
        legmode = mpatches.Patch(color="C4", label='Maximum of marginal posterior')

    ti_label = "Training ensemble"
    if hist_ti_ens == True:
        ti_hist_data = np.ravel(prior_ens)
        ti_hist_data = ti_hist_data[0.5*np.max(np.abs(ti_hist_data))>np.abs(ti_hist_data)]
    elif hist_ti_ens == "all":
        ti_hist_data = prior_ens
    else:
        ti_hist_data = seqsim_data
        ti_label = "Training model"

    # Training model
    if hist_ti_ens == "all":        
        for i in np.arange(0,np.shape(ti_hist_data)[1]):
            hist_ens_plot = ti_hist_data[:,[i]]
            y,binEdges=np.histogram(hist_ens_plot,bins=hist_bins, density = hist_density)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            if i == 0:
                ax.plot(bincenters,y, color = color_rgb, label=ti_label, linewidth = lwidth/lwidth_div,zorder=0.0)  
            else:
                ax.plot(bincenters,y, color = color_rgb, linewidth = lwidth/lwidth_div,zorder=0.0)
        leg3 = mpatches.Patch(color=color_rgb, label=ti_label)
    else:
        y,binEdges=np.histogram(ti_hist_data,bins=hist_bins, density = hist_density)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        ax.plot(bincenters,y, color = 'k', label=ti_label, linewidth = lwidth_mult*lwidth)
        leg3 = mpatches.Patch(color="k", label=ti_label)

    # Synthetic truth
    if truth_obj is not None:
        y,binEdges=np.histogram(truth_data, bins=hist_bins, density = hist_density)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        ax.plot(bincenters,y,'C2', label='Synthetic truth', linewidth = lwidth_mult*lwidth)
        leg4 = mpatches.Patch(color="C2", label="Synthetic truth")

    ax.set_title('({}) Histogram reproduction'.format(plot_letter))
    if hist_ti_ens_limit is not None:
        ax.set_xlim(left=hist_ti_ens_limit[0], right=hist_ti_ens_limit[1])

    if patch_legend == True:
        leg_handle = [leg1,leg3]
        if m_mode is not None:
            leg_handle.insert(1,legmode)
        if np.logical_or(hist_local_posterior == True, hist_pos_mean == True):
            leg_handle.insert(1,leg2)
        if truth_obj is not None:
            leg_handle.insert(2,leg4)
        if uncon_obj is not None:
            leg_handle.insert(0,leg0)
        if m_equiv_lsq is not None:
            leg_handle.append(leglsq)
        
        ax.legend(handles=leg_handle, numpoints=1, labelspacing=1, loc='best', fontsize=label_fontsize, frameon=False)
    else:
        ax.legend(loc='best', fontsize = label_fontsize)
    
    ax.set_xlabel('Field value {}'.format(unit_field))
    if hist_density == True:
        ax.set_ylabel('PDF')
    else:
        ax.set_ylabel('Count')

    #### SEMI-VARIOGRAM ####
    if sv_use == True:
        if res_use == False:
            plot_letter = "b"
        else:
            plot_letter = "c"

        ax = fig.add_subplot(gs[1, :])

        # Unconditional posterior / Prior
        if uncon_obj is not None:
            for i in np.arange(0,uncon_obj.N_sim):
                uncon_obj.sv_m_DSS(len(uncon_data), 1, uncon_posterior[:,[i]], uncon_obj.sort_d, uncon_obj.n_lags, uncon_obj.max_cloud)
                if i == 0:
                    ax.plot(uncon_obj.lags[:lags_use], uncon_obj.pics_m_DSS[:lags_use,0], color = color_rgb, label='Prior', linewidth = lwidth/lwidth_div, zorder=0)
                else:
                    ax.plot(uncon_obj.lags[:lags_use], uncon_obj.pics_m_DSS[:lags_use,0], color = color_rgb, linewidth = lwidth/lwidth_div, zorder=0) 
            leg0 = mpatches.Patch(color=color_rgb, label='Prior')

        # Posterior
        for i in np.arange(0,N_sim):
            seqsim_obj.sv_m_DSS(len(seqsim_data), 1, posterior_fields[:,[i]], seqsim_obj.sort_d, seqsim_obj.n_lags, seqsim_obj.max_cloud)
            if i == 0:
                ax.plot(seqsim_obj.lags[:lags_use], seqsim_obj.pics_m_DSS[:lags_use,0], color = color_rgb_zesty_pos, label='Posterior', linewidth = lwidth)
            else:
                ax.plot(seqsim_obj.lags[:lags_use], seqsim_obj.pics_m_DSS[:lags_use,0], color = color_rgb_zesty_pos, linewidth = lwidth) 
        leg1 = mpatches.Patch(color=color_rgb_zesty_pos, label='Posterior')

        # Model SV
        lags_use_max = np.max(seqsim_obj.lags[:lags_use])
        lags_use_idx_model = seqsim_obj.lags_sv_curve<=lags_use_max
        plot_model_lag = seqsim_obj.lags_sv_curve[lags_use_idx_model]
        plot_model_sv = seqsim_obj.sv_curve[lags_use_idx_model]
        ax.plot(plot_model_lag, plot_model_sv, color='k', label='Semi-variogram model', linewidth = lwidth_mult*lwidth) #linestyle = "dashed"
        leg2 = mpatches.Patch(color='k', linestyle = "-", label='Semi-variogram model')

        # Training model
        seqsim_obj.sv_m_DSS(len(seqsim_data), 1, seqsim_data.reshape(-1,1), seqsim_obj.sort_d, seqsim_obj.n_lags, seqsim_obj.max_cloud)
        ax.plot(seqsim_obj.lags[:lags_use], seqsim_obj.pics_m_DSS[:lags_use,0], "x", color = 'k', label='Training model', linewidth = lwidth_mult*lwidth) # marker = "x"
        #leg3 = mpatches.Patch(color="k", hatch = "x", label='Training model')
        #leg3 = mpatches.Circle((0,0), color="k", label='Training model')
        leg3 = mlines.Line2D([], [], color='k', linestyle = '', marker='x', markersize=8, label='Training model')

        if sv_pos_mean == True:
            # Realization mean
            seqsim_obj.sv_m_DSS(len(seqsim_data), 1, np.mean(posterior_fields,axis=1).reshape(-1,1), seqsim_obj.sort_d, seqsim_obj.n_lags, seqsim_obj.max_cloud)
            ax.plot(seqsim_obj.lags[:lags_use], seqsim_obj.pics_m_DSS[:lags_use,0], color = color_rgb_zesty_neg, label='Posterior mean', linewidth = lwidth_mult*lwidth)
            leg4 = mpatches.Patch(color=color_rgb_zesty_neg, label='Posterior mean')

        # Equivalent LSQ
        if m_equiv_lsq is not None:
            seqsim_obj.sv_m_DSS(len(seqsim_data), 1, np.array(m_equiv_lsq), seqsim_obj.sort_d, seqsim_obj.n_lags, seqsim_obj.max_cloud)
            ax.plot(seqsim_obj.lags[:lags_use], seqsim_obj.pics_m_DSS[:lags_use,0], color = 'C3', label='Equivalent LSQ', linewidth = lwidth_mult*lwidth)    

        # Observed truth
        if truth_obj is not None:
            seqsim_obj.sv_m_DSS(len(truth_data), 1, truth_data.reshape(-1,1), seqsim_obj.sort_d, seqsim_obj.n_lags, seqsim_obj.max_cloud)
            ax.plot(seqsim_obj.lags[:lags_use], seqsim_obj.pics_m_DSS[:lags_use,0], 'C2', label='Synthetic truth', linewidth = lwidth_mult*lwidth) 
            leg5 = mpatches.Patch(color="C2", label='Synthetic truth')

        ax.set_title('({}) Semi-variogram reproduction'.format(plot_letter))
        ax.set_ylabel('Semi-variance {}'.format(unit_var))
        ax.set_xlabel('Lag {}'.format(unit_lag))

        if patch_legend == True:
            leg_handle = [leg1,leg3,leg2]
            if sv_pos_mean == True:
                leg_handle.insert(1,leg4)
            if truth_obj is not None:
                leg_handle.insert(2,leg5)
            if uncon_obj is not None:
                leg_handle.insert(0,leg0)
            ax.legend(handles=leg_handle, numpoints=1, labelspacing=1, loc='best', fontsize=label_fontsize, frameon=False)
        else:
            ax.legend(loc='best', fontsize = label_fontsize)

    #### P-SPEC ####
    if spec_use == True:
        if sv_use == False:
            if res_use == False:
                plot_letter = "b"
            else:
                plot_letter = "c"
            ax = fig.add_subplot(gs[1, :])
        else:
            if res_use == False:
                plot_letter = "c"
            else:
                plot_letter = "d"
            ax = fig.add_subplot(gs[2, :])

        if seqsim_obj.grid_nmax < seqsim_obj.N_SH:
            nmax = seqsim_obj.grid_nmax
        else:
            nmax = seqsim_obj.N_SH
            
        ns = np.arange(1, nmax+1)
        n_ticks = np.append(np.array([1, 5, 10,]), np.arange(15,np.max(ns)+spec_step, step=spec_step))

        #N_ensembles = np.shape(seqsim_obj.g_spec)[-1]

        if  spec_r_at == None:
            spec_r_at = seqsim_obj.r_sat
        
        # Unconditional posterior / Prior
        if uncon_obj is not None:
            p_spec_pos_all = []
            for i in np.arange(0,uncon_obj.N_sim):
                ens_cilm = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,), uncon_obj.g_spec[:,i]))))
                if spec_mag == True:
                    p_spec_pos = pyshtools.gravmag.mag_spectrum(ens_cilm, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm)[1]))
                else:
                    p_spec_pos = pyshtools.spectralanalysis.spectrum(ens_cilm, degrees = np.arange(1,np.shape(ens_cilm)[1]))
                p_spec_pos = p_spec_pos[:nmax]
                p_spec_pos_all.append(p_spec_pos)
                if i == 0:
                    ax.plot(ns, p_spec_pos, color=color_rgb, label = "Prior", linewidth = lwidth/lwidth_div, zorder = 0.05)
                else:
                    ax.plot(ns, p_spec_pos, color=color_rgb, linewidth = lwidth/lwidth_div, zorder = 0.05) #color=color_rgb
            p_spec_pos_all = np.array(p_spec_pos_all)
            leg0 = mpatches.Patch(color=color_rgb, label='Prior')

        # Realizations
        p_spec_pos_all = []
        for i in np.arange(0,N_sim):
            ens_cilm = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,), seqsim_obj.g_spec[:,i]))))
            if spec_mag == True:
                p_spec_pos = pyshtools.gravmag.mag_spectrum(ens_cilm, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm)[1]))
            else:
                p_spec_pos = pyshtools.spectralanalysis.spectrum(ens_cilm, degrees = np.arange(1,np.shape(ens_cilm)[1]))
            p_spec_pos = p_spec_pos[:nmax]
            p_spec_pos_all.append(p_spec_pos)
            if i == 0:
                ax.plot(ns, p_spec_pos, color=color_rgb_zesty_pos, label = "Posterior", linewidth = lwidth, zorder = 0.05)
            else:
                ax.plot(ns, p_spec_pos, color=color_rgb_zesty_pos, linewidth = lwidth, zorder = 0.05) #color=color_rgb
        p_spec_pos_all = np.array(p_spec_pos_all)
        leg1 = mpatches.Patch(color=color_rgb_zesty_pos, label='Posterior')

        # Realization mean
        ens_cilm = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,), seqsim_obj.g_spec_mean))))
        #p_spec_pos_mean = pyshtools.gravmag.mag_spectrum(ens_cilm, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm)[1]))
        if spec_mag == True:
            p_spec_pos_mean = pyshtools.gravmag.mag_spectrum(ens_cilm, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm)[1]))
        else:
            p_spec_pos_mean = pyshtools.spectralanalysis.spectrum(ens_cilm, degrees = np.arange(1,np.shape(ens_cilm)[1]))
        p_spec_pos_mean = p_spec_pos_mean[:nmax]
        ax.plot(ns, p_spec_pos_mean, color=color_rgb_zesty_neg, label = "Posterior mean", linewidth = lwidth_mult*lwidth)
        leg2 = mpatches.Patch(color=color_rgb_zesty_neg, label='Posterior mean')

        # Equivalent LSQ
        if m_equiv_lsq is not None:
            ens_cilm_lsq = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,), seqsim_obj.g_equiv_lsq))))
            #p_spec_lsq = pyshtools.gravmag.mag_spectrum(ens_cilm_lsq, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm_lsq)[1])) # degrees to skip zeroth degree
            if spec_mag == True:
                p_spec_lsq = pyshtools.gravmag.mag_spectrum(ens_cilm_lsq, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm_lsq)[1]))
            else:
                p_spec_lsq = pyshtools.spectralanalysis.spectrum(ens_cilm_lsq, degrees = np.arange(1,np.shape(ens_cilm_lsq)[1]))
            p_spec_lsq = p_spec_lsq[:nmax]
            ax.plot(ns, p_spec_lsq, color = "C3", label = "Equivalent LSQ", linewidth = lwidth_mult*lwidth)
            leglsq = mpatches.Patch(color="C3", label='Equivalent LSQ')

        # Posterior Mode
        if m_mode is not None:
            ens_cilm = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,), seqsim_obj.g_spec_mode))))
            #p_spec_lsq = pyshtools.gravmag.mag_spectrum(ens_cilm_lsq, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm_lsq)[1])) # degrees to skip zeroth degree
            if spec_mag == True:
                p_spec_mode = pyshtools.gravmag.mag_spectrum(ens_cilm, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm)[1]))
            else:
                p_spec_mode = pyshtools.spectralanalysis.spectrum(ens_cilm, degrees = np.arange(1,np.shape(ens_cilm)[1]))
            p_spec_mode = p_spec_mode[:nmax]
            ax.plot(ns, p_spec_mode, color = "C4", label = "Maximum of marginal posterior", linewidth = lwidth_mult*lwidth)
            legmode = mpatches.Patch(color="C4", label='Maximum of marginal posterior')

        # Prior/Training
        if spec_ti_ens == True:
            n_max = seqsim_obj.N_SH

            # g ensemble and parameters
            if seqsim_obj.sim_type == "core_ens":
                g_ens = np.genfromtxt("mikkel_tools/models_shc/gnm_midpath.dat").T*10**9
                g_ens = g_ens[:shc_vec_len(seqsim_obj.N_SH),:]
                g_cut = g_ens[:n_max*(2+n_max),200:] # Truncate g
            elif seqsim_obj.sim_type == "lith_ens":
                g_ens = np.load("mikkel_tools/models_shc/lithosphere_g_in_rotated.npy")
                g_cut = g_ens[:shc_vec_len(n_max),::20]

            R = lowe_shspec(n_max,spec_r_at,seqsim_obj.a,g_cut)

            for i in np.arange(R.shape[1]):
                if i == 0:
                    ax.plot(ns,R[:,i], color = color_rgb, label = "Training ensemble", linewidth = lwidth/lwidth_div, zorder=0) #, linestyle = "dashed"
                    
                else:
                    ax.plot(ns,R[:,i], color = color_rgb, linewidth = lwidth/lwidth_div, zorder=0)
            leg3 = mpatches.Patch(color=color_rgb, label='Training ensemble')
        else:
            ens_cilm_prior = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,), seqsim_obj.g_prior))))
            #p_spec_prior = pyshtools.gravmag.mag_spectrum(ens_cilm_prior, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm_prior)[1])) # degrees to skip zeroth degree
            if spec_mag == True:
                p_spec_prior = pyshtools.gravmag.mag_spectrum(ens_cilm_prior, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm_prior)[1]))
            else:
                p_spec_prior = pyshtools.spectralanalysis.spectrum(ens_cilm_prior, degrees = np.arange(1,np.shape(ens_cilm_prior)[1]))
            p_spec_prior = p_spec_prior[:nmax]
            ax.plot(ns, p_spec_prior, color = "k", label = "Training model", linewidth = lwidth, zorder=0)
            leg3 = mpatches.Patch(color="k", label='Training model')

        # Observed truth
        if truth_obj is not None:
            ens_cilm_compare = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,), truth_obj.g_prior))))
            #p_spec_compare = pyshtools.gravmag.mag_spectrum(ens_cilm_compare, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm_compare)[1])) # degrees to skip zeroth degree
            if spec_mag == True:
                p_spec_compare = pyshtools.gravmag.mag_spectrum(ens_cilm_compare, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm_compare)[1]))
            else:
                p_spec_compare = pyshtools.spectralanalysis.spectrum(ens_cilm_compare, degrees = np.arange(1,np.shape(ens_cilm_compare)[1]))
            p_spec_compare = p_spec_compare[:nmax]
            ax.plot(ns, p_spec_compare, color = "C2", label = "Synthetic truth", linewidth = lwidth_mult*lwidth)
            leg4 = mpatches.Patch(color="C2", label='Synthetic truth')

        if spec_show_differences == True:
            # Differences
            #color_rgb_diff = (0.8,0.8,0.8)
            color_rgb_diff = "C8"
            for i in np.arange(N_sim):
                if i == 0:
                    ax.plot(ns, np.abs(p_spec_compare - p_spec_pos_all[i,:]), color=color_rgb_diff, label = "Truth - Posterior", linewidth = lwidth, zorder = 0)
                else:
                    ax.plot(ns, np.abs(p_spec_compare - p_spec_pos_all[i,:]), color=color_rgb_diff, linewidth = lwidth, zorder = 0)

            ax.plot(ns, np.abs(p_spec_compare - p_spec_pos_mean), color="C1", label = "Truth - Posterior mean", linewidth = lwidth, zorder = 0.1)
            ax.plot(ns, np.abs(p_spec_compare - p_spec_lsq), color = "C5", label = "Truth - Equivalent LSQ", linewidth = lwidth, linestyle = "dashed", zorder = 0.2)


        # Models
        if model_dict is not None: # Load models
            WDMAM2 = spio.loadmat('mikkel_tools/models_shc/WDMAM2.mat')
            LCS1 = load_shc("mikkel_tools/models_shc/LCS-1.shc")
            MF7 = load_shc("mikkel_tools/models_shc/MF7.shc")
            EMM2017 = np.loadtxt('mikkel_tools/models_shc/EMM2017.COF',comments="%",skiprows=1)

            # Add zero coefficient to comply with SHTOOLS methods
            g_LCS1 = np.hstack((np.zeros(1,),LCS1[:,2]))
            g_WDMAM2 = np.hstack((np.zeros(1,),WDMAM2["gh_wdmam"][:,0]))
            g_EMM2017 = np.hstack((np.zeros(1,),gauss_vector(EMM2017, 790, i_n = 2, i_m = 3)))

            # Also add "missing" coefficients for degree 0-15
            g_MF7 = np.hstack((np.zeros(shc_vec_len(15,include_n_zero = True),),MF7[:,2])) 

            # cilm
            cilm_LCS1 = pyshtools.shio.SHVectorToCilm(g_LCS1)
            cilm_MF7 = pyshtools.shio.SHVectorToCilm(g_MF7)
            cilm_WDMAM2 = pyshtools.shio.SHVectorToCilm(g_WDMAM2)
            cilm_EMM2017 = pyshtools.shio.SHVectorToCilm(g_EMM2017)

            # Pomme
            Gauss_in_pomme = np.loadtxt('mikkel_tools/models_shc/POMME_6_main_field.txt')
            g_pomme = np.hstack((np.zeros(1,), gauss_vector(Gauss_in_pomme, 60, i_n = 2, i_m = 3)))
            cilm_pomme = pyshtools.shio.SHVectorToCilm(g_pomme)

            # CHAOS 7
            N_chaos = 20
            CHAOS7 = load_CHAOS_matfile('mikkel_tools/models_shc/CHAOS-7.mat')
            chaos_time = mjd2000(spec_chaos_time[0], spec_chaos_time[1], spec_chaos_time[2])
            g_CHAOS7 = np.hstack((np.zeros(1,),CHAOS7.synth_coeffs_tdep(chaos_time, nmax=20, deriv=0)))
            cilm_CHAOS7 = pyshtools.shio.SHVectorToCilm(g_CHAOS7)
            model_dict_def = {"LCS-1":cilm_LCS1, "MF7":cilm_MF7, "WDMAM2":cilm_WDMAM2, "EMM2017":cilm_EMM2017, "POMME-6":cilm_pomme, "CHAOS-7":cilm_CHAOS7}

        if type(model_dict) is set or model_dict=="default":
            if model_dict=="default":
                model_dict = model_dict_def

            i = 0
            model_leg = []
            for key in model_dict:
                ens_cilm = model_dict_def[key]
                p_spec = pyshtools.gravmag.mag_spectrum(ens_cilm, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm)[1]))

                if key == "EMM2017" or key == "CHAOS-7":
                    use_ns = ns[:20]
                    use_p_spec = p_spec[:len(use_ns)]
                elif key == "MF7" or key == "LCS-1" or key == "WDMAM2":
                    use_ns = ns[16-1:]
                    use_p_spec = p_spec[16-1:nmax]
                elif key == "POMME-6":
                    use_ns = ns[:60]
                    use_p_spec = p_spec[:len(use_ns)]
                else:
                    use_ns = ns
                    use_p_spec = p_spec[:nmax]

                ax.plot(use_ns, use_p_spec, color="C{}".format(i+2), label = key, linewidth = lwidth)
                model_leg.append(mpatches.Patch(color="C{}".format(i+2), label=key))
                i += 1
        
        ax.set_yscale('log')
        ax.set_xlabel("degree, n")
        if spec_mag == True:
            ax.set_title('({}) Power spectra comparison, $r\'={}{}$'.format(plot_letter,spec_r_at,unit_lag[1:-1]))
            ax.set_ylabel("Power {}".format(unit_var))
        else:
            ax.set_title('({}) Power spectra comparison'.format(plot_letter))
            ax.set_ylabel("Power")    
        ax.set_xticks(n_ticks) #fontsize="small"
        ax.grid(alpha=0.3)

        if patch_legend == True:
            leg_handle_model = [leg1,leg2,leg3]
            if m_mode is not None:
                leg_handle_model.insert(2,legmode)
            if truth_obj is not None:
                leg_handle_model.append(leg4)
            if type(model_dict) is set or model_dict=="default":
                for leg in model_leg:
                    leg_handle_model.append(leg)
            if uncon_obj is not None:
                leg_handle_model.insert(0,leg0)
            if m_equiv_lsq is not None:
                leg_handle_model.append(leglsq)
            ax.legend(handles=leg_handle_model, numpoints=1, labelspacing=1, loc='best', fontsize=label_fontsize, frameon=False)
        else:
            ax.legend(loc='best', fontsize = label_fontsize)

    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    if savefig == True:
        fig.savefig('{}sdssim_reproduce_{}.{}'.format(save_path,save_string,save_ftype), bbox_inches='tight', dpi = save_dpi) 

    fig.show()
    

def plot_ensemble_map_tiles(lon, lat, ensemble_fields, field_uncon = None, field_compare = None, field_lsq = None, field_mean = None, tile_size_row = 3, tile_size_column = 3, figsize=(8,8), limit_for_SF = 10**6, point_size = 0.1,
                            left=0.03, bottom=0.12, right=0.97, top=0.95, wspace = 0.05, hspace=0.25, coast_width = 0.4, coast_color = "grey", cbar_mm_factor = 1, unit_transform_n_to_m = False,
                            savefig = False, save_string = "", save_dpi = 100,  save_path = "", save_ftype = "pdf", projection = ccrs.Mollweide(), cbar_limit = None, cbar_h = 0.1, cbar_text = "nT", cbar_text_color = "grey", use_gridlines = False, gridlines_width = 0.4, gridlines_alpha = 0.4):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tick
    import cartopy.crs as ccrs
    import matplotlib.colors as colors
    from matplotlib.colorbar import Colorbar

    if unit_transform_n_to_m == True:
        ensemble_fields = ensemble_fields*10**(-6)
        if field_mean is not None:
            field_mean = field_mean*10**(-6)
        if field_compare is not None:
            field_compare = field_compare*10**(-6)
            if field_lsq is not None:
                field_lsq = field_lsq*10**(-6)
            

    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0.0, 0.5, 1.0]
            return np.ma.masked_array(np.interp(value, x, y))

    SF = tick.ScalarFormatter() # Formatter for colorbar
    SF.set_powerlimits((6, 6)) # Set sci exponent used    

    fig = plt.figure(figsize=figsize, constrained_layout=False, dpi = save_dpi) # Initiate figure with constrained layout
    if field_uncon is None:
        fig.suptitle("Posterior realizations")

    # Generate ratio lists
    h_ratio = [1]*tile_size_row
    #h_ratio.append(cbar_h)
    w_ratio = [1]*tile_size_column

    if field_mean is None:
        h_ratio[-1] = cbar_h
        gs = fig.add_gridspec(tile_size_row, tile_size_column, height_ratios=h_ratio, width_ratios=w_ratio) # Add x-by-y grid
        #field_max = np.max(ensemble_fields)
        #field_min = np.min(ensemble_fields)
        field_max_true = np.max(ensemble_fields)
        field_min_true = np.min(ensemble_fields)
        field_max = cbar_mm_factor*np.max((abs(field_max_true),abs(field_min_true)))
        field_min = -field_max
    elif field_compare is None:
        h_ratio.append(1.5)
        h_ratio[-2] = cbar_h
        gs = fig.add_gridspec(tile_size_row+1, tile_size_column, height_ratios=h_ratio, width_ratios=w_ratio) # Add x-by-y grid
        #field_max = np.max(ensemble_fields)
        #field_min = np.min(ensemble_fields)
        field_max_true = np.max(ensemble_fields)
        field_min_true = np.min(ensemble_fields)
        field_max = cbar_mm_factor*np.max((abs(field_max_true),abs(field_min_true)))
        field_min = -field_max
    else:
        h_ratio.append(cbar_h)
        h_ratio.append(0.01)
        h_ratio.append(1.1)
        gs = fig.add_gridspec(tile_size_row+3, tile_size_column, height_ratios=h_ratio, width_ratios=w_ratio) # Add x-by-y grid

        field_max_true = np.max(field_compare)
        field_min_true = np.min(field_compare)
        field_max = cbar_mm_factor*np.max((abs(field_max_true),abs(field_min_true)))
        field_min = -field_max
    
    if cbar_limit is not None:
        field_max = np.max(cbar_limit)
        field_min = np.min(cbar_limit)

    ens_n = 0
    for i in np.arange(0,tile_size_row):
        for j in np.arange(0,tile_size_column):

            ax = fig.add_subplot(gs[i, j], projection=projection)
            ax.set_global()

            if np.logical_and.reduce((i == tile_size_row-1, j == 0, field_compare is not None, field_mean is not None)):
                plot_field = field_mean
                ax.set_title("Posterior mean")
            elif np.logical_and.reduce((i == tile_size_row-1, j == 0, field_compare is None, field_mean is not None)):
                ax = fig.add_subplot(gs[i+1, j], projection=projection)
                ax.set_global()
                plot_field = field_mean
                ax.set_title("Posterior mean")
            elif np.logical_and.reduce((i == tile_size_row-1, j == tile_size_column-1, field_compare is not None, field_mean is not None)):
                plot_field = np.std(ensemble_fields,axis=1)
                ax.set_title("Posterior std. deviation")
            elif np.logical_and.reduce((i == tile_size_row-1, j == tile_size_column-1, field_compare is None, field_mean is not None)):
                ax = fig.add_subplot(gs[i+1, j], projection=projection)
                ax.set_global()
                plot_field = np.std(ensemble_fields,axis=1)
                ax.set_title("Posterior std. deviation")
            elif np.logical_and.reduce((field_uncon is not None, i == 0, j == 0)):
                field_var = np.var(field_uncon,axis = 0) # Unconditional fields based on min/max variance
                idx_min_var = np.argmin(field_var)
                idx_max_var = np.argmax(field_var)
                plot_field = field_uncon[:,idx_min_var]
            elif np.logical_and.reduce((field_uncon is not None, i == 0, j == 1)):
                plot_field = field_uncon[:,idx_max_var]
                #plt.figtext(0.5,0.95,"________ Prior realizations ________", va="center", ha="center")
                #plt.figtext(0.5,0.8, "______ Posterior realizations ______", va="center", ha="center")
                plt.figtext(0.5,0.95,"Prior realizations", va="center", ha="center", fontsize = 12)
                plt.figtext(0.5,0.8, "Posterior realizations", va="center", ha="center", fontsize = 12)

            else:
                plot_field = ensemble_fields[:,ens_n]

            im = ax.scatter(lon, lat, s=point_size, c=plot_field, transform=ccrs.PlateCarree(), rasterized=True, vmin = field_min, vmax = field_max, cmap=cm_zesty_cbf, norm = MidpointNormalize(midpoint=0.))
            ax.coastlines(linewidth = coast_width, color = coast_color)
            ens_n += 1

            if use_gridlines == True:
                from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
                gl_lines = ax.gridlines(draw_labels=False,
                    linewidth=gridlines_width, color='black', alpha=gridlines_alpha, linestyle='-')
                gl_lines.xlines = True
                gl_lines.xlocator = tick.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
                gl_lines.ylocator = tick.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
                gl = ax.gridlines(alpha=0.0)

                gl.xlines = True
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.ylocator = tick.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
                
                if i == 0:
                    if j == 0:
                        gl.xlocator = tick.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135])
                    elif j == tile_size_column-1:
                        gl.xlocator = tick.FixedLocator([-135, -90, -45, 0, 45, 90, 135, 180])
                    gl.top_labels = True
                    gl.xlabel_style = {'size': 7, 'color': 'gray'}
                if j == 0:
                    gl.left_labels = True
                    gl.ylabel_style = {'size': 7, 'color': 'gray'}

    if field_compare is None:
        cbax = plt.subplot(gs[tile_size_row-1,:]) # Set colorbar position
    else:
        cbax = plt.subplot(gs[tile_size_row,:]) # Set colorbar position

    if field_max>limit_for_SF:
        cb = Colorbar(mappable = im, ax = cbax, orientation = "horizontal", format = SF) # im, ax=ax, 
    else:
        cb = Colorbar(mappable = im, ax = cbax, orientation = "horizontal")

    #cb.ax.text(0.5, -0.0, cbar_text, ha='center', va='center', color = cbar_text_color)
    cb.set_label(cbar_text)

    if field_compare is not None:
        if field_lsq is not None:
            for i in np.arange(0,tile_size_column):
                ax = fig.add_subplot(gs[-1, i], projection=projection)
                ax.set_global()

                if i == tile_size_column-1:
                    im = ax.scatter(lon, lat, s=point_size, c=field_compare, transform=ccrs.PlateCarree(), rasterized=True, vmin = field_min, vmax = field_max, cmap=cm_zesty_cbf, norm = MidpointNormalize(midpoint=0.))
                    #ax.annotate('Training model', (0.4, -0.1), xycoords='axes fraction', va='center')
                    ax.set_title("Synthetic truth")
                    #ax.set_xlabel("Training model")
                elif i == 0:
                    im = ax.scatter(lon, lat, s=point_size, c=field_lsq, transform=ccrs.PlateCarree(), rasterized=True, vmin = field_min, vmax = field_max, cmap=cm_zesty_cbf, norm = MidpointNormalize(midpoint=0.))
                    #ax.annotate('Equivalent LSQ', (0.4, -0.1), xycoords='axes fraction', va='center')
                    ax.set_title("Equivalent LSQ")
                    #ax.set_xlabel("Equivalent LSQ")

                ax.coastlines(linewidth = coast_width, color = coast_color)
                if use_gridlines == True:

                    gl = ax.gridlines(alpha=0.0)
                    #gl.top_labels = True
                    #gl.xlabel_style = {'size': 7, 'color': 'gray'}
                    #gl.left_labels = True
                    #gl.ylabel_style = {'size': 7, 'color': 'gray'}
                    gl.xlines = True
                    gl.xformatter = LONGITUDE_FORMATTER
                    gl.yformatter = LATITUDE_FORMATTER
                    #gl.xlocator = tick.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
                    gl.ylocator = tick.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

                    gl_lines = ax.gridlines(draw_labels=False,
                    linewidth=gridlines_width, color='black', alpha=gridlines_alpha, linestyle='-')
                    gl_lines.xlines = True
                    gl_lines.xlocator = tick.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
                    gl_lines.ylocator = tick.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

                    if i == 0:
                        gl.left_labels = True
                        gl.ylabel_style = {'size': 7, 'color': 'gray'}
        else:
            ax = fig.add_subplot(gs[-1, :], projection=projection)
            ax.set_global()
            im = ax.scatter(lon, lat, s=point_size, c=field_compare, transform=ccrs.PlateCarree(), rasterized=True, vmin = field_min, vmax = field_max, cmap=cm_zesty_cbf, norm = MidpointNormalize(midpoint=0.))
            ax.set_title("Synthetic truth")
            ax.coastlines(linewidth = coast_width, color = coast_color)
            if use_gridlines == True:

                gl = ax.gridlines(alpha=0.0)
                gl.xlines = True
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.ylocator = tick.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

                gl_lines = ax.gridlines(draw_labels=False,
                linewidth=gridlines_width, color='black', alpha=gridlines_alpha, linestyle='-')
                gl_lines.xlines = True
                gl_lines.xlocator = tick.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
                gl_lines.ylocator = tick.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

                gl.left_labels = True
                gl.ylabel_style = {'size': 7, 'color': 'gray'}


    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    if savefig == True:
        fig.savefig('{}map_tiles_{}.{}'.format(save_path, save_string, save_ftype), bbox_inches='tight', dpi = save_dpi) 

    fig.show()


def plot_map_compare_tiles(lon, lat, ul, ur, ll, lr, tile_size_row = 2, tile_size_column = 2, figsize=(8,8), limit_for_SF = 10**6, point_size = 0.1,
                            left=0.03, bottom=0.12, right=0.97, top=0.95, wspace = 0.05, hspace=0.25, coast_width = 0.4, coast_color = "grey", unit_transform_n_to_m = False,
                            savefig = False, save_string = "", save_ftype = "pdf", save_dpi = 100,  save_path = "", projection = ccrs.Mollweide(), project_ortho = False, cbar_limit = None, cbar_h = 0.1, cbar_text = "nT", 
                            cbar_text_color = "grey", use_gridlines = False, gridlines_width = 0.4, gridlines_alpha = 0.4):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tick
    import cartopy.crs as ccrs
    import matplotlib.colors as colors
    from matplotlib.colorbar import Colorbar
    import matplotlib.path as mpath

    if unit_transform_n_to_m == True:
        ul[1] = ul[1]*10**(-6)
        ur[1] = ur[1]*10**(-6)
        ll[1] = ll[1]*10**(-6)
        lr[1] = lr[1]*10**(-6)

    tile = [ul,ur,ll,lr]
    tile_letter = ["(a) ", "(b) ", "(c) ", "(d) "]

    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0.0, 0.5, 1.0]
            return np.ma.masked_array(np.interp(value, x, y))

    SF = tick.ScalarFormatter() # Formatter for colorbar
    SF.set_powerlimits((6, 6)) # Set sci exponent used    

    fig = plt.figure(figsize=figsize, constrained_layout=False, dpi = save_dpi) # Initiate figure with constrained layout
    #fig.suptitle("Posterior realizations")

    # Generate ratio lists
    h_ratio = [1]*tile_size_row
    h_ratio.append(cbar_h)
    w_ratio = [1]*tile_size_column
    
    gs = fig.add_gridspec(tile_size_row+1, tile_size_column, height_ratios=h_ratio, width_ratios=w_ratio) # Add x-by-y grid
    
    if cbar_limit is not None:
        field_max = np.max(cbar_limit)
        field_min = np.min(cbar_limit)
    else:
        field_max = np.max((abs(ul[1]),abs(ul[1])))
        field_min = -field_max

    lon_old = lon.copy()
    lat_old = lat.copy()
    for i in np.arange(0,len(tile)):
        if project_ortho == True:
            lon = lon_old[tile[i][4]]
            lat = lat_old[tile[i][4]]
            projection = ccrs.Orthographic(central_longitude=tile[i][2], central_latitude=tile[i][3])
            ax = fig.add_subplot(gs[i], projection=projection)
            theta_c = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta_c), np.cos(theta_c)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
        else:
            ax = fig.add_subplot(gs[i], projection=projection)
            ax.set_global()    

        ax.set_title(tile_letter[i]+tile[i][0])

        im = ax.scatter(lon, lat, s=point_size, c=tile[i][1], transform=ccrs.PlateCarree(), rasterized=True, vmin = field_min, vmax = field_max, cmap=cm_zesty_cbf, norm = MidpointNormalize(midpoint=0.))
        ax.coastlines(linewidth = coast_width, color = coast_color)
        

        if use_gridlines == True:
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
            gl_lines = ax.gridlines(draw_labels=False,
                linewidth=gridlines_width, color='black', alpha=gridlines_alpha, linestyle='-')
            gl_lines.xlines = True
            gl_lines.xlocator = tick.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
            gl_lines.ylocator = tick.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
            gl = ax.gridlines(alpha=0.0)

            gl.xlines = True
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlocator = tick.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135])
            #gl.xlocator = tick.FixedLocator([-135, -90, -45, 0, 45, 90, 135, 180])
            gl.ylocator = tick.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
            gl.top_labels = True
            gl.xlabel_style = {'size': 7, 'color': 'gray'}
            if np.logical_or(i == 0, i == 2):
                gl.left_labels = True
                gl.ylabel_style = {'size': 7, 'color': 'gray'}

    cbax = plt.subplot(gs[tile_size_row,:]) # Set colorbar position

    if field_max>limit_for_SF:
        cb = Colorbar(mappable = im, ax = cbax, orientation = "horizontal", format = SF) # im, ax=ax, 
    else:
        cb = Colorbar(mappable = im, ax = cbax, orientation = "horizontal")
    cb.set_label(cbar_text)


    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    if savefig == True:
        fig.savefig('{}map_compare_tiles_{}.{}'.format(save_path, save_string, save_ftype), bbox_inches='tight', dpi = save_dpi) 

    fig.show()


def plot_global(lat = None, lon = None, data=None, limits_data = None, cbar_use = True,
                cbar_h = 0.075, cbar_even = True, cbar_mm_factor = 1, cbar_text = "", cbar_text_color = "grey",
                unit_transform_n_to_m = False, projection_transformation = "Mollweide", figsize=(6,6),
                coast_width = 0.4, coast_color = "grey", limit_for_SF = 10**6, extent = None,
                left=0.03, bottom=0.35, right=0.97, top=0.95, wspace = 0.05, hspace=0.01,
                title="", lat_0 = 0.0, lon_0 = 0.0, point_size=1, save_bg_color = "w",
                savefig = False, save_dpi = 100, save_string ="", save_path = "", save_ftype = "pdf", save_transparent = False, rasterize = True,
                use_gridlines = False, gridlines_width = 0.4, gridlines_alpha = 0.4,
                data_on_top = False, color_bg = None, color_bg_transparent = False, color_points = "C0", cmap = None, midnorm = 0.0):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tick
    import cartopy.crs as ccrs
    import matplotlib.colors as colors 
    from matplotlib.colorbar import Colorbar

    # Colormap
    if cmap is None:
        cmap = cm_zesty_cbf

    # Plotting init
    if projection_transformation == "ortho":
        projection = ccrs.Orthographic(central_longitude=lon_0, central_latitude=lat_0)
    elif projection_transformation == "Miller":
        projection = ccrs.Miller()
    elif projection_transformation == "PlateCarree":
        projection = ccrs.PlateCarree()
    else:
        projection = ccrs.Mollweide(central_longitude=0)

    if unit_transform_n_to_m == True:
        data = data*10**(-6)
        if limits_data is not None:
            if len(limits_data)>2:
                limits_data = limits_data*10**(-6)            

    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0.0, 0.5, 1.0]
            return np.ma.masked_array(np.interp(value, x, y))

    SF = tick.ScalarFormatter() # Formatter for colorbar
    SF.set_powerlimits((6, 6)) # Set sci exponent used    

    fig = plt.figure(figsize=figsize, constrained_layout=False, dpi = save_dpi) # Initiate figure with constrained layout
    fig.suptitle("{}".format(title))

    # Generate ratio lists
    h_ratio = [1]
    h_ratio.append(cbar_h)
    w_ratio = [1]

    # Gridspec
    gs = fig.add_gridspec(2, 1, height_ratios=h_ratio, width_ratios=w_ratio) # Add x-by-y grid

    ax = fig.add_subplot(gs[0, 0], projection=projection)
    ax.set_global()
    #ax.set_title("{}".format(title))

    ax.coastlines(linewidth = coast_width, color = coast_color)

    if data is None:
        im = ax.scatter(lon, lat, s=point_size, color=color_points, marker = "o", transform=ccrs.PlateCarree(), rasterized=True, zorder = 10)
    else:
        # Plotting ranges and norms
        if not all(np.logical_or(type(element) == str, type(element) == np.str_)  for element in data):
            if limits_data is None:
                field_max_true = np.max(data)
                field_min_true = np.min(data)
            else:
                field_max_true = np.max(limits_data)
                field_min_true = np.min(limits_data)

            if cbar_even == True:
                field_max = cbar_mm_factor*np.max((abs(field_max_true),abs(field_min_true)))
                field_min = -field_max
            else:
                field_max = cbar_mm_factor*field_max_true
                field_min = cbar_mm_factor*field_min_true
        else:
            field_min = None
            field_max = None

        # Plotting
        if data_on_top == True:
            im = ax.scatter(lon, lat, s=point_size, c=data, marker = "o", transform=ccrs.PlateCarree(), rasterized=True, vmin = field_min, vmax = field_max, cmap=cmap, norm = MidpointNormalize(midpoint=midnorm), zorder = 10)
        else:
            im = ax.scatter(lon, lat, s=point_size, c=data, marker = "o", transform=ccrs.PlateCarree(), rasterized=True, vmin = field_min, vmax = field_max, cmap=cmap, norm = MidpointNormalize(midpoint=midnorm))

        
    #im = ax.imshow(data, transform=ccrs.PlateCarree(), cmap=cm_zesty_cbf, vmin = field_min, vmax = field_max, norm = MidpointNormalize(midpoint=0.))
    
    
    #ax.stock_img()
    if color_bg is not None:
        ax.background_patch.set_facecolor(color_bg)
    if color_bg_transparent == True:
        fig.patch.set_alpha(0)
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    if use_gridlines == True:
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        gl_lines = ax.gridlines(draw_labels=False,
            linewidth=gridlines_width, color='black', alpha=gridlines_alpha, linestyle='-')
        gl_lines.xlines = True
        if extent is not None:
            gl_lines.ylocator = tick.FixedLocator(np.arange(-90,90+10,10))
            gl_lines.xlocator = tick.FixedLocator(np.arange(-180,180+10,10))
        else:
            gl_lines.ylocator = tick.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
            gl_lines.xlocator = tick.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        gl = ax.gridlines(alpha=0.0)

        gl.xlines = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if extent is not None:
            gl.ylocator = tick.FixedLocator(np.arange(-90,90+10,10))
            gl.xlocator = tick.FixedLocator(np.arange(-180,180+10,10))
        else:
            gl.ylocator = tick.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
            gl.xlocator = tick.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        gl.top_labels = True
        gl.xlabel_style = {'size': 7, 'color': 'gray'}
        gl.left_labels = True
        gl.ylabel_style = {'size': 7, 'color': 'gray'}

    if cbar_use == True:
        cbax = plt.subplot(gs[1,:]) # Set colorbar position

        if field_max>limit_for_SF:
            cb = Colorbar(mappable = im, ax = cbax, orientation = "horizontal", format = SF) # im, ax=ax, 
        else:
            cb = Colorbar(mappable = im, ax = cbax, orientation = "horizontal")

        cb.set_label(cbar_text)
        
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    if savefig == True:
        fig.savefig('{}map_{}.{}'.format(save_path, save_string, save_ftype), bbox_inches='tight', dpi = save_dpi, facecolor=save_bg_color, edgecolor=save_bg_color, transparent = save_transparent) 

    fig.show()


def plot_local_dist_KL(zs_DSS, skip = 1, N_bins = 21, idx_high_start = -401, idx_high_interval = 100, idx_low_start = 0, idx_low_end = 1250, idx_low_interval = 250, 
                       xlim = [-2,2], hist_density = False, save_dpi = 100, savefig = False, save_path = "", save_string = "", figsize = (11,8), random_seed = None, leg_size = "small"):
    from scipy import stats
    from mpl_toolkits import mplot3d

    if random_seed is not None:
        np.random.seed(random_seed)

    zs_sort = zs_DSS[np.argsort(np.mean(zs_DSS,axis=1)),:]

    skip = 1
    m_skip = zs_sort[:,:][0::skip]
    m_skip_mean = np.mean(m_skip,axis=1)
    m_skip_std = np.std(m_skip,axis=1)
    m_normal = np.random.normal(size=m_skip.shape) * m_skip_std[:,None] + m_skip_mean[:,None] # Generate equivalent normal distributions for each local posterior

    # Histograms
    m_centers = []
    m_y = []
    for i in np.arange(0,m_skip.shape[0]):
        y,binEdges=np.histogram(m_skip[i,:],bins=N_bins,density=hist_density)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        m_centers.append(bincenters)
        m_y.append(y)
    m_centers = np.array(m_centers)
    m_y = np.array(m_y)

    m_centers_n = []
    m_y_n = []
    for i in np.arange(0,m_skip.shape[0]):
        y,binEdges=np.histogram(m_normal[i,:],bins=N_bins,density=hist_density)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        m_centers_n.append(bincenters)
        m_y_n.append(y)
    m_centers_n = np.array(m_centers_n)
    m_y_n = np.array(m_y_n)

    # Compute Kullback-Leibler divergence between local posterior and equivalent normal distributions
    kld = stats.entropy(m_y_n.T+10**(-30), m_y.T+10**(-30))
    idx_kld_all = np.argsort(kld)

    # PLOTTING
    fig = plt.figure(figsize=figsize, constrained_layout=False, dpi = save_dpi) # Initiate figure with constrained layout
    gs = fig.add_gridspec(2, 2, height_ratios=[2,1], width_ratios=[1,1]) # Add x-by-y grid

    # HIGH SAMPLES
    idx_bc_high = np.arange(0,len(kld))[idx_high_start::idx_high_interval]
    idx_kld_high = idx_kld_all[idx_bc_high]
    ax = fig.add_subplot(gs[0, 1], projection="3d")

    x_c = m_centers[idx_kld_high,:].T*10**(-6)
    z_y = m_y[idx_kld_high,:].T
    y_kld = np.repeat(kld[None,idx_kld_high],x_c.shape[0],axis=0)
    x_c_n = m_centers_n[idx_kld_high,:].T*10**(-6)
    z_y_n = m_y_n[idx_kld_high,:].T

    x_c[x_c>xlim[1]]= np.nan
    x_c[x_c<xlim[0]]= np.nan

    for i in np.arange(0,len(idx_bc_high)):
        ax.plot3D(x_c[:,i], y_kld[:,i], z_y[:,i], color=color_rgb_zesty_pos, zorder = 1/(idx_bc_high[i]+10**(-3)))
        ax.plot3D(x_c_n[:,i], y_kld[:,i], z_y_n[:,i], linestyle="dashed", color="k", zorder = 1/(idx_bc_high[i]+10**(-3)))

    ax.set_zlabel("Count")
    ax.set_ylabel("KL-divergence")
    ax.set_xlabel("Field value [mT]")
    ax.set_xlim(xlim)
    ax.set_xticks([-2,-1,0,1,1,2])
    ax.grid(b=None)
    ax.view_init(30, -40)
    leg1 = mpatches.Patch(color=color_rgb_zesty_pos, label="Sampled marginal posterior")
    leg2 = mpatches.Patch(color="k", label="Equivalent Gaussian")
    ax.legend(handles=[leg1,leg2], numpoints=1, 
            labelspacing=1, loc='best', fontsize=leg_size, frameon=False)

    # LOW SAMPLES
    idx_bc_low = np.arange(0,len(kld))[idx_low_start:idx_low_end:idx_low_interval]
    idx_kld_low = idx_kld_all[idx_bc_low]
    ax = fig.add_subplot(gs[0, 0], projection="3d")

    x_c = m_centers[idx_kld_low,:].T*10**(-6)
    z_y = m_y[idx_kld_low,:].T
    y_kld = np.repeat(kld[None,idx_kld_low],x_c.shape[0],axis=0)
    x_c_n = m_centers_n[idx_kld_low,:].T*10**(-6)
    z_y_n = m_y_n[idx_kld_low,:].T

    x_c[x_c>xlim[1]]= np.nan
    x_c[x_c<xlim[0]]= np.nan

    for i in np.arange(0,len(idx_bc_low)):
        ax.plot3D(x_c[:,i], y_kld[:,i], z_y[:,i], color=color_rgb_zesty_neg, zorder = 1/(idx_bc_low[i]+10**(-3)))
        ax.plot3D(x_c_n[:,i], y_kld[:,i], z_y_n[:,i], linestyle="dashed", color="k", zorder = 1/(idx_bc_low[i]+10**(-3)))

    ax.set_zlabel("Count")
    ax.set_ylabel("KL-divergence")
    ax.set_xlabel("Field value [mT]")
    ax.set_xlim(xlim)
    ax.set_xticks([-2,-1,0,1,1,2])
    ax.grid(b=None)
    ax.view_init(30, -40)
    leg1 = mpatches.Patch(color=color_rgb_zesty_neg, label="Sampled marginal posterior")
    leg2 = mpatches.Patch(color="k", label="Equivalent Gaussian")
    ax.legend(handles=[leg1,leg2], numpoints=1, 
            labelspacing=1, loc='best', fontsize=leg_size, frameon=False)

    # KL LINE PLOT
    ax = fig.add_subplot(gs[1, :])
    ax.semilogy(kld[idx_kld_all],color=(0.6,0.6,0.6))
    ax.vlines(idx_bc_low, -2, kld[idx_kld_low], colors=color_rgb_zesty_neg, zorder = 10)
    ax.vlines(idx_bc_high, -2, kld[idx_kld_high], colors=color_rgb_zesty_pos, zorder = 10)
    ax.set_ylabel("Kullback-Leibler divergence")
    ax.set_xlabel("Sorted model parameter index")
    leg1 = mpatches.Patch(color=color_rgb_zesty_neg, label="Samples from low KL-divergence")
    leg2 = mpatches.Patch(color=color_rgb_zesty_pos, label="Samples from high KL-divergence")
    leg3 = mpatches.Patch(color=(0.6,0.6,0.6), label="KL-divergence")
    ax.legend(handles=[leg1,leg2,leg3], numpoints=1, 
            labelspacing=1, loc='best', fontsize=leg_size, frameon=False)

    if savefig == True:
        fig.savefig('{}local_dist_KL_{}.pdf'.format(save_path,save_string), bbox_inches='tight', dpi = save_dpi) 

    fig.show()


def plot_spec_compare(nmax, spec_r_at, spec_r_ref, g1, g2, g3, g4, truth = None, model_dict = None,
                      figsize=(8,8), title = "Power spectra comparison", sim_type = "core_ens", spec_step = 5, lwidth = 1, lwidth_mult = 2, lwidth_div = 5,
                      left=0.03, bottom=0.12, right=0.97, top=0.95, wspace = 0.05, hspace=0.25, label_fontsize = "small",
                      savefig = False, save_string = "", save_dpi = 100,  save_path = ""):
    import numpy as np
    import matplotlib.pyplot as plt
    color_rgb = (0.6,0.6,0.6)
    tile = [g1, g2, g3, g4]

    fig = plt.figure(figsize=figsize, constrained_layout=False, dpi = save_dpi) # Initiate figure with constrained layout

    # Generate ratio lists
    h_ratio = [1,1]
    w_ratio = [1]
    gs = fig.add_gridspec(2, 1, height_ratios=h_ratio, width_ratios=w_ratio) # Add x-by-y grid
    ax = fig.add_subplot(gs[0])    
    ax2 = fig.add_subplot(gs[1])

    #### P-SPEC ####
    ns = np.arange(1, nmax+1)
    n_ticks = np.append(np.array([1, 5, 10,]), np.arange(15,np.max(ns)+spec_step, step=spec_step))
    leg_handle = []
    R_truth = lowe_shspec(nmax,spec_r_at,spec_r_ref,truth)

    # Realizations
    p_spec_pos_all = []
    for i in np.arange(0,len(tile)):
        R = lowe_shspec(nmax,spec_r_at,spec_r_ref,tile[i][1])
        ax.plot(ns, R, color="C{}".format(i), linewidth = lwidth, zorder = tile[i][2]) #color=color_rgb
        ax2.plot(ns, np.mean(abs(R_truth-R),axis=1), color="C{}".format(i), linewidth = lwidth_mult*lwidth, zorder = tile[i][2]) #color=color_rgb
        leg_handle.append(mpatches.Patch(color="C{}".format(i), label=tile[i][0]))

    # Observed truth
    if truth is not None:
        ax.plot(ns, R_truth, color = "k", label = "Synthetic truth", linewidth = lwidth_mult*lwidth,zorder=10)
        leg_handle.append(mpatches.Patch(color="k", label='Synthetic truth'))

    # Prior/Training
    # g ensemble and parameters
    if sim_type == "core_ens":
        g_ens = np.genfromtxt("mikkel_tools/models_shc/gnm_midpath.dat").T*10**9
        g_ens = g_ens[:shc_vec_len(nmax),:]
        g_cut = g_ens[:,200:] # Truncate g
    elif sim_type == "lith_ens":
        g_ens = np.load("mikkel_tools/models_shc/lithosphere_g_in_rotated.npy")
        g_cut = g_ens[:shc_vec_len(n_max),::20]

    R_prior = lowe_shspec(nmax,spec_r_at,spec_r_ref,g_cut)
    ax.plot(ns,R_prior, color = color_rgb, linewidth = lwidth/lwidth_div, zorder=0)
    leg_handle.append(mpatches.Patch(color=color_rgb, label='Training ensemble'))

    # Models
    if model_dict is not None: # Load models
        WDMAM2 = spio.loadmat('mikkel_tools/models_shc/WDMAM2.mat')
        LCS1 = load_shc("mikkel_tools/models_shc/LCS-1.shc")
        MF7 = load_shc("mikkel_tools/models_shc/MF7.shc")
        EMM2017 = np.loadtxt('mikkel_tools/models_shc/EMM2017.COF',comments="%",skiprows=1)

        # Add zero coefficient to comply with SHTOOLS methods
        g_LCS1 = np.hstack((np.zeros(1,),LCS1[:,2]))
        g_WDMAM2 = np.hstack((np.zeros(1,),WDMAM2["gh_wdmam"][:,0]))
        g_EMM2017 = np.hstack((np.zeros(1,),gauss_vector(EMM2017, 790, i_n = 2, i_m = 3)))

        # Also add "missing" coefficients for degree 0-15
        g_MF7 = np.hstack((np.zeros(shc_vec_len(15,include_n_zero = True),),MF7[:,2])) 

        # cilm
        cilm_LCS1 = pyshtools.shio.SHVectorToCilm(g_LCS1)
        cilm_MF7 = pyshtools.shio.SHVectorToCilm(g_MF7)
        cilm_WDMAM2 = pyshtools.shio.SHVectorToCilm(g_WDMAM2)
        cilm_EMM2017 = pyshtools.shio.SHVectorToCilm(g_EMM2017)

        # Pomme
        Gauss_in_pomme = np.loadtxt('mikkel_tools/models_shc/POMME_6_main_field.txt')
        g_pomme = np.hstack((np.zeros(1,), gauss_vector(Gauss_in_pomme, 60, i_n = 2, i_m = 3)))
        cilm_pomme = pyshtools.shio.SHVectorToCilm(g_pomme)

        # CHAOS 7
        N_chaos = 20
        CHAOS7 = load_CHAOS_matfile('mikkel_tools/models_shc/CHAOS-7.mat')
        chaos_time = mjd2000(spec_chaos_time[0], spec_chaos_time[1], spec_chaos_time[2])
        g_CHAOS7 = np.hstack((np.zeros(1,),CHAOS7.synth_coeffs_tdep(chaos_time, nmax=20, deriv=0)))
        cilm_CHAOS7 = pyshtools.shio.SHVectorToCilm(g_CHAOS7)
        model_dict_def = {"LCS-1":cilm_LCS1, "MF7":cilm_MF7, "WDMAM2":cilm_WDMAM2, "EMM2017":cilm_EMM2017, "POMME-6":cilm_pomme, "CHAOS-7":cilm_CHAOS7}

    if type(model_dict) is set or model_dict=="default":
        if model_dict=="default":
            model_dict = model_dict_def

        i = 0
        model_leg = []
        for key in model_dict:
            ens_cilm = model_dict_def[key]
            p_spec = pyshtools.gravmag.mag_spectrum(ens_cilm, spec_r_ref, spec_r_at, degrees = np.arange(1,np.shape(ens_cilm)[1]))

            if key == "EMM2017" or key == "CHAOS-7":
                use_ns = ns[:20]
                use_p_spec = p_spec[:len(use_ns)]
            elif key == "MF7" or key == "LCS-1" or key == "WDMAM2":
                use_ns = ns[16-1:]
                use_p_spec = p_spec[16-1:nmax]
            elif key == "POMME-6":
                use_ns = ns[:60]
                use_p_spec = p_spec[:len(use_ns)]
            else:
                use_ns = ns
                use_p_spec = p_spec[:nmax]

            ax.plot(use_ns, use_p_spec, color="C{}".format(i+4), label = key, linewidth = lwidth)
            leg_handle.append(mpatches.Patch(color="C{}".format(i+4), label=key))
            i += 1

    ax.set_yscale('log')
    ax.set_xlabel("degree, n")
    ax.set_title('{}, $r\'={}km$'.format(title, spec_r_at))
    ax.set_ylabel("Power [nT²]")   
    ax.set_xticks(n_ticks) #fontsize="small"
    ax.grid(alpha=0.3)
    ax.legend(handles=leg_handle, numpoints=1, labelspacing=1, loc='best', fontsize=label_fontsize, frameon=False)

    ax2.set_yscale('log')
    ax2.set_xlabel("degree, n")
    ax2.set_title('Mean absolute difference to synthetic truth')
    ax2.set_ylabel("Power [nT²]")   
    ax2.set_xticks(n_ticks) #fontsize="small"
    ax2.grid(alpha=0.3)
    ax2.legend(handles=leg_handle[:-2], numpoints=1, labelspacing=1, loc='best', fontsize=label_fontsize, frameon=False)


    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    if savefig == True:
        fig.savefig('{}spec_compare_{}.pdf'.format(save_path, save_string), bbox_inches='tight', dpi = save_dpi) 

    fig.show()


def plot_power_spectrum(p_spec, figsize=(14,8)):
    import matplotlib.pyplot as plt
    import numpy as np
    ns = np.arange(1,len(p_spec))
    n_ticks = np.append(np.append(np.array([1,]),np.arange(10,np.max(ns),step=10)),np.max(ns))
    plt.figure(figsize=figsize)
    plt.plot(ns, p_spec[1:])
    plt.yscale('log')
    plt.xlabel("Spherical harmonic degree")
    plt.ylabel("Power [nT²]")
    plt.xticks(n_ticks, fontsize="small")
    plt.grid(alpha=0.3)
    plt.show()

def plot_p_spec(g_spec, p_spec_height, nmax, model_dict = None, figsize = (14,8), lwidth = 2, lwidth_m = 2, step = 5, spec_style = None, 
                label = "ensemble", color = "lightgray", legend_loc = "best", r_ref = 6371.2, g_spec_compares = None, nmax_pairs = None, nmax_pairs_compare = None, label_compare = None, color_compare = (0,0,0),
                savefig = False, save_string = ""):

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy as sp
    import scipy.io as spio
    import pyshtools

    # ChaosMagPy modules
    from chaosmagpy import load_CHAOS_matfile
    from chaosmagpy.model_utils import synth_values
    from chaosmagpy.data_utils import mjd2000

    fig = plt.figure(figsize=figsize)

    ns = np.arange(1,nmax+1)
    n_ticks = np.append(np.array([1, 5, 10,]),np.arange(15,np.max(ns)+step,step=step))

    if spec_style == "pair_compare":
        rgb_gradient = 0.6
        for g_spec_use, g_spec_compare, nmax_use, nmax_compare, labels in zip(g_spec, g_spec_compares, nmax_pairs, nmax_pairs_compare, label):
            ens_cilm = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,),g_spec_use))))
            ens_cilm_compare = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,),g_spec_compare))))

            p_spec = pyshtools.gravmag.mag_spectrum(ens_cilm, r_ref, p_spec_height, degrees = np.arange(1,np.shape(ens_cilm)[1])) # degrees to skip zeroth degree
            p_spec_compare = pyshtools.gravmag.mag_spectrum(ens_cilm_compare, r_ref, p_spec_height, degrees = np.arange(1,np.shape(ens_cilm_compare)[1])) # degrees to skip zeroth degree
            
            p_spec = p_spec[:nmax_use]
            p_spec_compare = p_spec_compare[:nmax_compare]

            labels_use = labels + " estimate"
            labels_compare = labels + " label"

            #plt.plot(np.arange(1,nmax_use+1), p_spec, color=(rgb_gradient, rgb_gradient, rgb_gradient), label = labels_use, linewidth = lwidth)
            #plt.plot(np.arange(1,nmax_compare+1), p_spec_compare, color=(rgb_gradient, rgb_gradient, rgb_gradient), label = labels_compare, linewidth = lwidth, linestyle = "dashed")
            plt.plot(np.arange(1,nmax_use+1), p_spec, color=(0.6,0.6,0.6), label = labels_use, linewidth = lwidth)
            plt.plot(np.arange(1,nmax_compare+1), p_spec_compare, color=(0.2,0.2,0.2), label = labels_compare, linewidth = lwidth, linestyle = "dashed")

            #rgb_gradient -= 0.4

    elif spec_style == "ensemble":
        N_ensembles = np.shape(g_spec)[-1]

        for i in np.arange(0,N_ensembles):
            ens_cilm = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,),g_spec[:,i]))))
            p_spec = pyshtools.gravmag.mag_spectrum(ens_cilm, r_ref, p_spec_height, degrees = np.arange(1,np.shape(ens_cilm)[1]))
            p_spec = p_spec[:nmax]
            if i == 0:
                plt.plot(ns, p_spec, color=color, label = label, linewidth = lwidth)
            else:
                plt.plot(ns, p_spec, color=color, linewidth = lwidth)

        if g_spec_compares is not None:
            for i in np.arange(0,g_spec_compares.shape[1]):
                ens_cilm_compare = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,),g_spec_compares[:,i]))))
                p_spec_compare = pyshtools.gravmag.mag_spectrum(ens_cilm_compare, r_ref, p_spec_height, degrees = np.arange(1,np.shape(ens_cilm_compare)[1])) # degrees to skip zeroth degree
                p_spec_compare = p_spec_compare[:nmax]
                if i == 0:
                    plt.plot(ns, p_spec_compare, color=color_compare, label = label_compare, linewidth = lwidth, linestyle = "dashed")
                else:
                    plt.plot(ns, p_spec_compare, color=color_compare, linewidth = lwidth, linestyle = "dashed")

    else:
        N_ensembles = np.shape(g_spec)[-1]

        for i in np.arange(0,N_ensembles):
            ens_cilm = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,),g_spec[:,i]))))
            p_spec = pyshtools.gravmag.mag_spectrum(ens_cilm, r_ref, p_spec_height, degrees = np.arange(1,np.shape(ens_cilm)[1]))
            p_spec = p_spec[:nmax]
            if i == 0:
                plt.plot(ns, p_spec[:nmax], color=color, label = label, linewidth = lwidth)
            else:
                plt.plot(ns, p_spec[:nmax], color=color, linewidth = lwidth)

    if model_dict is not None: # Load models
        WDMAM2 = spio.loadmat('mikkel_tools/models_shc/WDMAM2.mat')
        LCS1 = load_shc("mikkel_tools/models_shc/LCS-1.shc")
        MF7 = load_shc("mikkel_tools/models_shc/MF7.shc")
        EMM2017 = np.loadtxt('mikkel_tools/models_shc/EMM2017.COF',comments="%",skiprows=1)

        # Add zero coefficient to comply with SHTOOLS methods
        g_LCS1 = np.hstack((np.zeros(1,),LCS1[:,2]))
        g_WDMAM2 = np.hstack((np.zeros(1,),WDMAM2["gh_wdmam"][:,0]))
        g_EMM2017 = np.hstack((np.zeros(1,),gauss_vector(EMM2017, 790, i_n = 2, i_m = 3)))

        # Also add "missing" coefficients for degree 0-15
        g_MF7 = np.hstack((np.zeros(shc_vec_len(15,include_n_zero = True),),MF7[:,2])) 

        # cilm
        cilm_LCS1 = pyshtools.shio.SHVectorToCilm(g_LCS1)
        cilm_MF7 = pyshtools.shio.SHVectorToCilm(g_MF7)
        cilm_WDMAM2 = pyshtools.shio.SHVectorToCilm(g_WDMAM2)
        cilm_EMM2017 = pyshtools.shio.SHVectorToCilm(g_EMM2017)

        # Pomme
        Gauss_in_pomme = np.loadtxt('mikkel_tools/models_shc/POMME_6_main_field.txt')
        g_pomme = np.hstack((np.zeros(1,), gauss_vector(Gauss_in_pomme, 60, i_n = 2, i_m = 3)))
        cilm_pomme = pyshtools.shio.SHVectorToCilm(g_pomme)

        # CHAOS 7
        N_chaos = 20
        CHAOS7 = load_CHAOS_matfile('mikkel_tools/models_shc/CHAOS-7.mat')
        chaos_time = mjd2000(2020, 1, 1)
        g_CHAOS7 = np.hstack((np.zeros(1,),CHAOS7.synth_coeffs_tdep(chaos_time, nmax=20, deriv=0)))
        cilm_CHAOS7 = pyshtools.shio.SHVectorToCilm(g_CHAOS7)
        model_dict_def = {"LCS-1":cilm_LCS1, "MF7":cilm_MF7, "WDMAM2":cilm_WDMAM2, "EMM2017":cilm_EMM2017, "POMME-6":cilm_pomme, "CHAOS-7":cilm_CHAOS7}

    if type(model_dict) is set or model_dict=="default":
        if model_dict=="default":
            model_dict = model_dict_def

        i = 0
        for key in model_dict:
            ens_cilm = model_dict_def[key]
            p_spec = pyshtools.gravmag.mag_spectrum(ens_cilm, r_ref, p_spec_height, degrees = np.arange(1,np.shape(ens_cilm)[1]))

            if key == "EMM2017" or key == "CHAOS-7":
                use_ns = ns[:20]
                use_p_spec = p_spec[:len(use_ns)]
            elif key == "MF7" or key == "LCS-1" or key == "WDMAM2":
                use_ns = ns[16-1:]
                use_p_spec = p_spec[16-1:nmax]
            elif key == "POMME-6":
                use_ns = ns[:60]
                use_p_spec = p_spec[:len(use_ns)]
            else:
                use_ns = ns
                use_p_spec = p_spec[:nmax]

            plt.plot(use_ns, use_p_spec, color="C{}".format(i), label = key, linewidth = lwidth_m)
            i += 1

    plt.yscale('log')
    plt.xlabel("degree n")
    plt.ylabel("Power [nT²]")
    plt.xticks(n_ticks, fontsize="small")
    plt.grid(alpha=0.3)
    plt.legend(loc = legend_loc)

    if savefig == True:
        fig.savefig('p_spec_{}.pdf'.format(save_string), bbox_inches='tight') 

    fig.show()


def plot_ensemble_histogram(ensemble, N_ensemble, target = None, figsize=(10,10), unit = "", savefig = False, savepath = "./", filename = "file", fontsize = 10, dpi = 100):
    import numpy as np
    plt.figure(figsize=figsize)
    if N_ensemble > 1:
        for j in range(0,N_ensemble-1):
            y,binEdges=np.histogram(ensemble[:,j],bins=200)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            plt.plot(bincenters,y,'-',color = '0.75')
    else:
        j = -1
        
    y,binEdges=np.histogram(ensemble[:,j+1],bins=200)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters,y,'-',color = '0.75',label='Ensemble')    
    
    if target is not None:
        y,binEdges=np.histogram(target,bins=200)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,'k-',label='Target')

    plt.legend(loc='best',fontsize=fontsize)
    plt.xlabel('Ensemble value {}'.format(unit), fontsize=fontsize, labelpad=8)
    plt.ylabel('Bin count', fontsize=fontsize, labelpad=8)
    if savefig == True: 
        plt.savefig('{}{}.png'.format(savepath, filename), bbox_inches='tight', dpi = dpi)
    plt.show()


def plot_CLiP_data(key, ens_s_sat, ens_Li, ens_C, labels, transform_to_map = False, shape_s = None, shape_Li = None, shape_C = None, figsize=(16,10)):
    # Easy random plot synth sat, lithosphere, core

    import matplotlib.pyplot as plt
    import matplotlib.ticker as tick

    if transform_to_map == True:
        plot_s = ens_s_sat[:,key].reshape(shape_s)
        plot_Li = ens_Li[:,labels[key][0]].reshape(shape_Li)
        plot_C = ens_C[:,labels[key][1]].reshape(shape_C)
    else:
        plot_s = ens_s_sat[key,:,:]
        plot_Li = ens_Li[labels[key][0],:,:]
        plot_C = ens_C[labels[key][1],:,:]


    SF = tick.ScalarFormatter() # Formatter for colorbar
    SF.set_powerlimits((3, 3)) # Set sci exponent used

    fig = plt.figure(figsize=figsize, constrained_layout=True) # Initiate figure with constrained layout
    gs = fig.add_gridspec(2, 2) # Add 2x2 grid

    ax1 = fig.add_subplot(gs[0, :]) # Use full row
    ax1.set_title('Synth sat obs, B_r [nT]')
    im1 = ax1.imshow(plot_s, cmap = cm_zesty_cbf)
    fig.colorbar(im1, ax=ax1, orientation = "horizontal", shrink=0.3, format = SF)

    ax2 = fig.add_subplot(gs[1, 0]) # Use one 
    ax2.set_title('Lithosphere, B_r [nT]')
    im2 = ax2.imshow(plot_Li, cmap = cm_zesty_cbf)
    fig.colorbar(im2, ax=ax2, orientation = "horizontal", shrink=0.6)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('Core, B_r [nT]')
    im3 = ax3.imshow(plot_C, cmap = cm_zesty_cbf)
    fig.colorbar(im3, ax=ax3, orientation = "horizontal", shrink=0.6, format = SF)
    plt.show()


def interpolate_grid(grid_in_theta, grid_out_theta, grid_in_phi, grid_out_phi, grid_in, method_int = "nearest", output = "return", save_path = ""):
    import scipy.interpolate as itp
    # Define interpolation grids
    grid_in_tuple = (90-np.ravel(grid_in_theta), np.ravel(grid_in_phi))
    grid_out_tuple = (90-np.ravel(grid_out_theta), np.ravel(grid_out_phi))

    # Interpolate grid
    grid_out = itp.griddata(grid_in_tuple, np.ravel(grid_in), grid_out_tuple, method=method_int)
    
    # Save or return
    if output == "return":
        return grid_out
    elif output == "save":
        np.save(save_path, grid_out)


def haversine(radius, lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    import numpy as np
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = radius * c
    return km


def array_nm(nmax):
    import numpy as np
    # Generate (degree,order) pairs
    N_SH_m_len = np.sum(np.arange(1,nmax+1)+1) # Number of (degree,order) pairs
    m_max = np.arange(0,nmax+1) # Orders for degree N
    m = np.zeros(N_SH_m_len) # Pre-allocate orders
    n = np.zeros(N_SH_m_len) # Pre-allocate degrees
    len_m_in = 0 # initiate count of inserted order lengths
    for i in range(0,nmax): # walk through degrees - 1
        m_in = m_max[:(i+2)] # Get orders for current degree, i+1
        i_m_in = m_in + len_m_in # Determine index for insertion
        m[i_m_in] = m_in # Insert orders for current degree, i+1
        n[i_m_in] = i+1 # Insert current degree
        len_m_in += len(m_in) # Update insertion index for next iteration

    nm = np.hstack((n.reshape(-1,1),m.reshape(-1,1))) # Collect (degree,order) pairs

    return nm


def lowe_shspec(n,c,a,g):
    import numpy as np
    # Compute Lowes spherical harmonic power spectrum up to degree n at radius c given reference a and coefficient vector g

    g_cut = g[:n*(2+n)] # Truncate g
    i = np.arange(1,n+1) # Degree array
    reps = (i*(2+i)-(i-1)*(1+i)) # Coefficients per degree
    ns = np.repeat(i,reps) # Repeat degrees for each corresponding coefficient
    const = (ns+1)*(a/c)**(2*ns+4) # Array of constants for each corresponding coefficient
    g_const_sq = const.reshape(-1,1)*np.power(g_cut,2) # Multiply constant with squared g
    rep_cs = np.cumsum(reps) # Cumsum to split array for each degree
    split = np.split(g_const_sq,rep_cs)[:-1] # Perform split using cumsum

    R = np.array([sum(i) for i in split]) # Sum coefficients for each degree and return numpy array
    
    return R


def gauss_vector(g_in, N_deg, i_n = 0, i_m = 1):
    # Function for computing a vector of Gauss coefficicents given standard input
    import numpy as np

    i=0
    i_line=0

    g = np.zeros(2*np.sum(np.arange(1,N_deg+1)+1) - N_deg)

    for n in range(1,N_deg+1):
        for m in range(0,n+1):
            if m == 0: 
                g[i]=g_in[i_line,i_n]
                i += 1
                i_line += 1            
            else:
                g[i]=g_in[i_line,i_n]
                g[i+1]=g_in[i_line,i_m]
                i+= 2  
                i_line += 1

    return g 


def gauss_vector_zeroth(g_in, N_deg, i_n = 0, i_m = 1):
    # Function for computing a vector of Gauss coefficicents given standard input
    import numpy as np

    i=0
    i_line=0

    g = np.zeros(2*np.sum(np.arange(1,N_deg+1)+1) - N_deg + 1)

    for n in range(0,N_deg+1):
        for m in range(0,n+1):
            if m == 0: 
                g[i]=g_in[i_line,i_n]
                i += 1
                i_line += 1            
            else:
                g[i]=g_in[i_line,i_n]
                g[i+1]=g_in[i_line,i_m]
                i+= 2  
                i_line += 1

    return g


def sh_expand_glq(glq_field, glq_nmax, glq_w, glq_zero, set_nmax, set_norm = 1, geomag_scale = True, geomag_ref = 6371.2, geomag_r_at = 3480.0):
    import pyshtools
    import numpy as np

    if geomag_scale == True:
        # Schmidt semi-normalized
        C_cilm = pyshtools.expand.SHExpandGLQ(glq_field.reshape(glq_nmax+1,2*glq_nmax+1), glq_w, glq_zero, [2, 1, glq_nmax])
        C_index = np.transpose(pyshtools.shio.SHCilmToCindex(C_cilm))

        nm_C = array_nm(glq_nmax)
        C_corr_sh = 1/(nm_C[:,[0]]+1)*1/(geomag_ref/geomag_r_at)**(nm_C[:,[0]]+2)
        C_index = C_index[1:,:]*C_corr_sh
        C_vec = gauss_vector(C_index, set_nmax, i_n = 0, i_m = 1)
    else:
        C_cilm = pyshtools.expand.SHExpandGLQ(glq_field.reshape(glq_nmax+1,2*glq_nmax+1), glq_w, glq_zero, [set_norm, 1, set_nmax])
        #C_index = np.transpose(pyshtools.shio.SHCilmToCindex(C_cilm))
        #C_vec = gauss_vector_zeroth(C_index, set_nmax, i_n = 0, i_m = 1)
        C_vec = pyshtools.shio.SHCilmToVector(C_cilm)
    
    return C_vec, C_cilm


def sh_makegrid_glq(C_vec, glq_nmax, glq_zero, set_nmax, set_norm = 1):        
    import pyshtools

    C_cilm = pyshtools.shio.SHVectorToCilm(C_vec)

    gridglq = pyshtools.expand.MakeGridGLQ(C_cilm, glq_zero, [glq_nmax, set_norm, 1, set_nmax, False])

    return gridglq


def shc_vec_len(nmax, nmin = 1, include_n_zero = False, include_zeros = False):
    """
    Returns number of spherical harmonic coefficients up to degree nmax with possible leading truncation before degree nmin
    - nmax: Maximum shc degree
    - nmin: Minimum shc degree
    - include_n_zero: Possibility of adding degree zero (required by functionality in e.g. SHTOOLS)
    - include_zeros: Include the zero value present for each order = 0
    """
    import numpy as np

    flag_n_zero = False

    #vec_len = np.sum(np.arange(1,nmax+1)*2+1)
    vec_len = (nmax - nmin + 1)*(nmax + nmin + 1)

    if include_n_zero == True:
        vec_len += 1
        flag_n_zero = True

    if include_zeros == True:
        vec_len += nmax-nmin+1
        if flag_n_zero == True:
            vec_len += 1

    return vec_len


def plot_clip_loss(epoch, train_loss, valid_loss, train_L_Li, train_L_C, valid_L_Li, valid_L_C, figsize=(10,5)):
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, constrained_layout=True) # Initiate figure with constrained layout
    gs = fig.add_gridspec(3, 2) # Add 3x2 grid

    ax1 = fig.add_subplot(gs[0, :]) # Use full row
    ax1.set_title("Loss")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean of batch loss')
    #ax1.plot(np.arange(epoch+1), train_loss, color="black")
    #ax1.plot(np.arange(epoch+1), valid_loss, color="gray", linestyle="--")
    ax1.semilogy(np.arange(epoch)+1, np.abs(np.array(train_loss)), color="black")
    ax1.semilogy(np.arange(epoch)+1, np.abs(np.array(valid_loss)), color="gray", linestyle="--")
    #ax1.set_ylim(np.abs(np.array(train_loss))[-1]*0.5, np.abs(np.array(train_loss))[0]*1.5)
    ax1.legend(['Training', 'Validation'])

    ax4 = fig.add_subplot(gs[1, 0])
    #ax5.semilogy(np.arange(epoch)+1, np.mean(train_L_Li,axis=1), color="C0")
    ax4.semilogy(np.arange(epoch)+1, train_L_Li, color="C0")
    ax4.set_title("Training lithosphere loss")
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Mean batch loss')

    ax5 = fig.add_subplot(gs[1, 1])
    #ax5.semilogy(np.arange(epoch)+1, np.mean(train_L_C,axis=1), color="C1")
    ax5.semilogy(np.arange(epoch)+1, train_L_C, color="C1")
    ax5.set_title("Training core loss")
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Mean batch loss')

    ax6 = fig.add_subplot(gs[2, 0])
    #ax6.semilogy(np.arange(epoch)+1, np.mean(valid_L_Li,axis=1), color="C0")
    ax6.semilogy(np.arange(epoch)+1, valid_L_Li, color="C0")
    ax6.set_title("Validation lithosphere loss")
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('mean batch loss')

    ax7 = fig.add_subplot(gs[2, 1])
    #ax6.semilogy(np.arange(epoch)+1, np.mean(valid_L_C,axis=1), color="C1")
    ax7.semilogy(np.arange(epoch)+1, valid_L_C, color="C1")
    ax7.set_title("Validation core loss")
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Mean batch loss')

    plt.show()

def plot_clip_grid_comparison(epoch_i, Li_out, C_out, sat_in, batch_labels, clip, map_shape = False, equal_amp = False, show_diff = False, figsize=(8,8), limit_for_SF = 10**6, shrink_factor = 1.0):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tick
    import cartopy.crs as ccrs
    import matplotlib.colors as colors

    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0.0, 0.5, 1.0]
            return np.ma.masked_array(np.interp(value, x, y))

    size_lat_in_Li = clip.ens_Li_shape[0]
    size_lon_in_Li = clip.ens_Li_shape[1]
    size_lat_in_C = clip.ens_C_shape[0]
    size_lon_in_C = clip.ens_C_shape[1]
    size_lat_in_s = clip.ens_s_shape[0]
    size_lon_in_s = clip.ens_s_shape[1]

    size_lat_out_Li = clip.ens_Li.shape[1]
    size_lon_out_Li = clip.ens_Li.shape[2]
    size_lat_out_C = clip.ens_C.shape[1]
    size_lon_out_C = clip.ens_C.shape[2]

    if map_shape == False:
        Li_in_plot = clip.ens_Li[:,batch_labels[epoch_i][0]].reshape((size_lon_out_Li,size_lat_out_Li)).T
        C_in_plot = clip.ens_C[:,batch_labels[epoch_i][1]].reshape((size_lon_out_C,size_lat_out_C)).T
        Li_out_plot = Li_out[epoch_i].reshape((size_lon_out_Li,size_lat_out_Li)).T
        C_out_plot = C_out[epoch_i].reshape((size_lon_out_C,size_lat_out_C)).T
    elif map_shape == "deconv":
        Li_in_plot = clip.ens_Li[batch_labels[epoch_i][0],:,:]
        C_in_plot = clip.ens_C[batch_labels[epoch_i][1],:,:]
        Li_out_plot = Li_out[epoch_i][0]
        C_out_plot = C_out[epoch_i][0]
        #print(np.array(Li_out).shape)
    else:
        Li_in_plot = clip.ens_Li[batch_labels[epoch_i][0],:,:]
        C_in_plot = clip.ens_C[batch_labels[epoch_i][1],:,:]
        Li_out_plot = Li_out[epoch_i].reshape((size_lat_out_Li,size_lon_out_Li))
        C_out_plot = C_out[epoch_i].reshape((size_lat_out_C,size_lon_out_C))

    batch_sat_plot = sat_in[epoch_i]

    if clip.normalize == 1:
        Li_in_plot = Li_in_plot*(clip.Li_scale[0]-clip.Li_scale[1])+clip.Li_scale[1]
        C_in_plot = C_in_plot*(clip.C_scale[0]-clip.C_scale[1])+clip.C_scale[1]
        Li_out_plot = Li_out_plot*(clip.Li_scale[0]-clip.Li_scale[1])+clip.Li_scale[1]
        C_out_plot = C_out_plot*(clip.C_scale[0]-clip.C_scale[1])+clip.C_scale[1]
        batch_sat_plot = batch_sat_plot*(clip.clip_scale[0]-clip.clip_scale[1])+clip.clip_scale[1]
        #sat_plot = sat_plot*(clip.clip_scale[0]-clip.clip_scale[1])+clip.clip_scale[1]
    if clip.normalize == 0:
        Li_in_plot = Li_in_plot*(clip.Li_scale[0]-clip.Li_scale[1])/2+clip.Li_scale[1]+(clip.Li_scale[0]-clip.Li_scale[1])/2
        C_in_plot = C_in_plot*(clip.C_scale[0]-clip.C_scale[1])/2+clip.C_scale[1]+(clip.C_scale[0]-clip.C_scale[1])/2
        Li_out_plot = Li_out_plot*(clip.Li_scale[0]-clip.Li_scale[1])/2+clip.Li_scale[1]+(clip.Li_scale[0]-clip.Li_scale[1])/2
        C_out_plot = C_out_plot*(clip.C_scale[0]-clip.C_scale[1])/2+clip.C_scale[1]+(clip.C_scale[0]-clip.C_scale[1])/2
        batch_sat_plot = batch_sat_plot*(clip.clip_scale[0]-clip.clip_scale[1])/2+clip.clip_scale[1]+(clip.clip_scale[0]-clip.clip_scale[1])/2
        #sat_plot = sat_plot*(clip.clip_scale[0]-clip.clip_scale[1])/2+clip.clip_scale[1]+(clip.clip_scale[0]-clip.clip_scale[1])/2
    if clip.normalize == "mean":
        Li_in_plot = Li_in_plot*(clip.Li_scale[0]-clip.Li_scale[1])+clip.Li_mean
        C_in_plot = C_in_plot*(clip.C_scale[0]-clip.C_scale[1])+clip.C_mean
        Li_out_plot = Li_out_plot*(clip.Li_scale[0]-clip.Li_scale[1])+clip.Li_mean
        C_out_plot = C_out_plot*(clip.C_scale[0]-clip.C_scale[1])+clip.C_mean
        batch_sat_plot = batch_sat_plot*(clip.clip_scale[0]-clip.clip_scale[1])+clip.clip_mean
    if clip.normalize == "standardized":
        Li_in_plot = Li_in_plot*clip.Li_std+clip.Li_mean
        C_in_plot = C_in_plot*clip.C_std+clip.C_mean
        Li_out_plot = Li_out_plot*clip.Li_std+clip.Li_mean
        C_out_plot = C_out_plot*clip.C_std+clip.C_mean
        batch_sat_plot = batch_sat_plot*clip.clip_std+clip.clip_mean

    if map_shape == "deconv":
        clip.clip_to_obs(Li_out_plot, C_out_plot, r_at = clip.r_sat)
        sat_plot = clip.B_clip_pred[:,0]
    else:
        clip.clip_to_obs(Li_out_plot, C_out_plot, r_at = clip.r_sat)
        sat_plot = clip.B_clip_pred[:,0]

    SF = tick.ScalarFormatter() # Formatter for colorbar
    SF.set_powerlimits((6, 6)) # Set sci exponent used    

    fig = plt.figure(figsize=figsize, constrained_layout=True) # Initiate figure with constrained layout

    if show_diff == True:
        gs = fig.add_gridspec(3, 3) # Add 3x2 grid
    else:
        gs = fig.add_gridspec(3, 2) # Add 3x2 grid


    print(batch_sat_plot.shape)
    ax01 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax01.set_global()
    ax01.set_title('Input synthetic sat')
    im01 = ax01.imshow(batch_sat_plot.reshape((size_lat_in_s,size_lon_in_s)), norm = MidpointNormalize(midpoint=0.), cmap = cm_zesty_cbf, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90])

    ax02 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()) 
    ax02.set_global()
    ax02.set_title('Output sat estimate')

    ax2 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()) 
    ax2.set_global()
    ax2.set_title('Output Lithosphere')
    
    ax3 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree()) 
    ax3.set_global()
    ax3.set_title('Output Core')

    ax4 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax4.set_global()  
    ax4.set_title('Label Lithosphere')
    im4 = ax4.imshow(Li_in_plot, norm = MidpointNormalize(midpoint=0.), cmap = cm_zesty_cbf, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90])
    
    ax5 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
    ax5.set_global()
    ax5.set_title('Label Core')
    im5 = ax5.imshow(C_in_plot, norm = MidpointNormalize(midpoint=0.), cmap = cm_zesty_cbf, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90])

    if show_diff == True:
        ax6 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
        ax6.set_global()
        ax6.set_title('Sat differences')

        ax7 = fig.add_subplot(gs[1, 2], projection=ccrs.PlateCarree())
        ax7.set_global()
        ax7.set_title('Lithosphere differences')

        ax8 = fig.add_subplot(gs[2, 2], projection=ccrs.PlateCarree())
        ax8.set_global()
        ax8.set_title('Core differences')



    if equal_amp == True:
        vmin_Li=np.min(Li_in_plot)
        vmax_Li=np.max(Li_in_plot)

        vmin_C=np.min(C_in_plot)
        vmax_C=np.max(C_in_plot)

        vmin_sat=np.min(batch_sat_plot)
        vmax_sat=np.max(batch_sat_plot)
    else:
        vmin_Li=None
        vmax_Li=None

        vmin_C=None
        vmax_C=None

        vmin_sat=None
        vmax_sat=None

    #if map_shape == False:
    #    im02 = ax02.imshow(sat_plot.reshape((size_lat_in,size_lon_in)), norm = MidpointNormalize(midpoint=0.), cmap = plt.cm.RdBu_r, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90], vmin = vmin_sat, vmax = vmax_sat)
    #    im2 = ax2.imshow(Li_out_plot, norm = MidpointNormalize(midpoint=0.), cmap = plt.cm.RdBu_r, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90], vmin = vmin_Li, vmax = vmax_Li)
    #    im3 = ax3.imshow(C_out_plot, norm = MidpointNormalize(midpoint=0.), cmap = plt.cm.RdBu_r, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90], vmin = vmin_C, vmax = vmax_C)
    #
    #    im6 = ax02.imshow(batch_sat_plot.reshape((size_lat_in,size_lon_in))-sat_plot.reshape((size_lat_in,size_lon_in)), norm = MidpointNormalize(midpoint=0.), cmap = plt.cm.RdBu_r, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90], vmin = vmin_sat, vmax = vmax_sat)
    #    im7 = ax2.imshow(Li_in_plot-Li_out_plot, norm = MidpointNormalize(midpoint=0.), cmap = plt.cm.RdBu_r, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90], vmin = vmin_Li, vmax = vmax_Li)
    #    im8 = ax3.imshow(C_in_plot-C_out_plot, norm = MidpointNormalize(midpoint=0.), cmap = plt.cm.RdBu_r, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90], vmin = vmin_C, vmax = vmax_C)
    #
    #else:
    im02 = ax02.imshow(sat_plot.reshape((size_lat_in_s,size_lon_in_s)), norm = MidpointNormalize(midpoint=0.), cmap = cm_zesty_cbf, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90], vmin = vmin_sat, vmax = vmax_sat)
    im2 = ax2.imshow(Li_out_plot, norm = MidpointNormalize(midpoint=0.), cmap = cm_zesty_cbf, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90], vmin = vmin_Li, vmax = vmax_Li)
    im3 = ax3.imshow(C_out_plot, norm = MidpointNormalize(midpoint=0.), cmap = cm_zesty_cbf, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90], vmin = vmin_C, vmax = vmax_C)
    
    if show_diff == True:
        im6 = ax6.imshow(batch_sat_plot.reshape((size_lat_in_s,size_lon_in_s))-sat_plot.reshape((size_lat_in_s,size_lon_in_s)), norm = MidpointNormalize(midpoint=0.), cmap = plt.cm.RdBu_r, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90])
        im7 = ax7.imshow(Li_in_plot-Li_out_plot, norm = MidpointNormalize(midpoint=0.), cmap = cm_zesty_cbf, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90])
        im8 = ax8.imshow(C_in_plot-C_out_plot, norm = MidpointNormalize(midpoint=0.), cmap = cm_zesty_cbf, transform=ccrs.PlateCarree(), extent=[-180, 180, 90, -90])
        #ax6.coastlines()
        #ax7.coastlines()
        #ax8.coastlines()

    #ax01.coastlines()
    #ax02.coastlines()
    #ax2.coastlines()
    #ax3.coastlines()
    #ax4.coastlines()
    #ax5.coastlines()

    
    #limit_for_SF = 10**3
    #shrink_factor = 1.0
    if np.max(batch_sat_plot)>limit_for_SF:
        fig.colorbar(im01, ax=ax01, orientation = "horizontal", shrink=shrink_factor, format = SF)
        fig.colorbar(im02, ax=ax02, orientation = "horizontal", shrink=shrink_factor, format = SF)
        if show_diff == True:
            fig.colorbar(im6, ax=ax6, orientation = "horizontal", shrink=shrink_factor, format = SF)
    else:
        fig.colorbar(im01, ax=ax01, orientation = "horizontal", shrink=shrink_factor)
        fig.colorbar(im02, ax=ax02, orientation = "horizontal", shrink=shrink_factor)
        if show_diff == True:
            fig.colorbar(im6, ax=ax6, orientation = "horizontal", shrink=shrink_factor)

    if np.max(Li_in_plot)>limit_for_SF:
        fig.colorbar(im2, ax=ax2, orientation = "horizontal", shrink=shrink_factor, format = SF)
        fig.colorbar(im4, ax=ax4, orientation = "horizontal", shrink=shrink_factor, format = SF)
        if show_diff == True:
            fig.colorbar(im7, ax=ax7, orientation = "horizontal", shrink=shrink_factor, format = SF)
    else:
        fig.colorbar(im2, ax=ax2, orientation = "horizontal", shrink=shrink_factor)
        fig.colorbar(im4, ax=ax4, orientation = "horizontal", shrink=shrink_factor)
        if show_diff == True:
            fig.colorbar(im7, ax=ax7, orientation = "horizontal", shrink=shrink_factor)

    if np.max(C_in_plot)>limit_for_SF:
        fig.colorbar(im3, ax=ax3, orientation = "horizontal", shrink=shrink_factor, format = SF)
        fig.colorbar(im5, ax=ax5, orientation = "horizontal", shrink=shrink_factor, format = SF)
        if show_diff == True:
            fig.colorbar(im8, ax=ax8, orientation = "horizontal", shrink=shrink_factor, format = SF)
    else:
        fig.colorbar(im3, ax=ax3, orientation = "horizontal", shrink=shrink_factor)
        fig.colorbar(im5, ax=ax5, orientation = "horizontal", shrink=shrink_factor)
        if show_diff == True:
            fig.colorbar(im8, ax=ax8, orientation = "horizontal", shrink=shrink_factor)      

    plt.show()


def plot_clip_grid_residuals(epoch_i, Li_out, C_out, sat_in, batch_labels, clip, map_shape = False, clip_at_sat = False, bins = 100, figsize = (12,8)):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tick

    size_lat_in = clip.grid_shape[1]
    size_lon_in = clip.grid_shape[0]
    size_lat_out_Li = clip.ens_Li.shape[1]
    size_lon_out_Li = clip.ens_Li.shape[2]
    size_lat_out_C = clip.ens_C.shape[1]
    size_lon_out_C = clip.ens_C.shape[2]

    #Li_in_plot = clip.ens_Li[:,batch_labels[epoch_i][0]].reshape((size_lon_out,size_lat_out)).T
    #C_in_plot = clip.ens_C[:,batch_labels[epoch_i][1]].reshape((size_lon_out,size_lat_out)).T

    #Li_out_plot = Li_out[epoch_i].reshape((size_lon_out,size_lat_out)).T
    #C_out_plot = C_out[epoch_i].reshape((size_lon_out,size_lat_out)).T

    if map_shape == False:
        Li_in_plot = clip.ens_Li[:,batch_labels[epoch_i][0]].reshape((size_lon_out_Li,size_lat_out_Li)).T
        C_in_plot = clip.ens_C[:,batch_labels[epoch_i][1]].reshape((size_lon_out_C,size_lat_out_C)).T
        Li_out_plot = Li_out[epoch_i].reshape((size_lon_out_Li,size_lat_out_Li)).T
        C_out_plot = C_out[epoch_i].reshape((size_lon_out,size_lat_out)).T
    else:
        Li_in_plot = clip.ens_Li[batch_labels[epoch_i][0],:,:]
        C_in_plot = clip.ens_C[batch_labels[epoch_i][1],:,:]
        Li_out_plot = Li_out[epoch_i].reshape((size_lat_out_Li,size_lon_out_Li))
        C_out_plot = C_out[epoch_i].reshape((size_lat_out_C,size_lon_out_C))

    batch_sat_plot = sat_in[epoch_i]

    if clip.normalize == 1:
        Li_in_plot = Li_in_plot*(clip.Li_scale[0]-clip.Li_scale[1])+clip.Li_scale[1]
        C_in_plot = C_in_plot*(clip.C_scale[0]-clip.C_scale[1])+clip.C_scale[1]
        Li_out_plot = Li_out_plot*(clip.Li_scale[0]-clip.Li_scale[1])+clip.Li_scale[1]
        C_out_plot = C_out_plot*(clip.C_scale[0]-clip.C_scale[1])+clip.C_scale[1]
        batch_sat_plot = batch_sat_plot*(clip.clip_scale[0]-clip.clip_scale[1])+clip.clip_scale[1]
        #sat_plot = sat_plot*(clip.clip_scale[0]-clip.clip_scale[1])+clip.clip_scale[1]
    if clip.normalize == 0:
        Li_in_plot = Li_in_plot*(clip.Li_scale[0]-clip.Li_scale[1])/2+clip.Li_scale[1]+(clip.Li_scale[0]-clip.Li_scale[1])/2
        C_in_plot = C_in_plot*(clip.C_scale[0]-clip.C_scale[1])/2+clip.C_scale[1]+(clip.C_scale[0]-clip.C_scale[1])/2
        Li_out_plot = Li_out_plot*(clip.Li_scale[0]-clip.Li_scale[1])/2+clip.Li_scale[1]+(clip.Li_scale[0]-clip.Li_scale[1])/2
        C_out_plot = C_out_plot*(clip.C_scale[0]-clip.C_scale[1])/2+clip.C_scale[1]+(clip.C_scale[0]-clip.C_scale[1])/2
        batch_sat_plot = batch_sat_plot*(clip.clip_scale[0]-clip.clip_scale[1])/2+clip.clip_scale[1]+(clip.clip_scale[0]-clip.clip_scale[1])/2
        #sat_plot = sat_plot*(clip.clip_scale[0]-clip.clip_scale[1])/2+clip.clip_scale[1]+(clip.clip_scale[0]-clip.clip_scale[1])/2
    if clip.normalize == "mean":
        Li_in_plot = Li_in_plot*(clip.Li_scale[0]-clip.Li_scale[1])+clip.Li_mean
        C_in_plot = C_in_plot*(clip.C_scale[0]-clip.C_scale[1])+clip.C_mean
        Li_out_plot = Li_out_plot*(clip.Li_scale[0]-clip.Li_scale[1])+clip.Li_mean
        C_out_plot = C_out_plot*(clip.C_scale[0]-clip.C_scale[1])+clip.C_mean
        batch_sat_plot = batch_sat_plot*(clip.clip_scale[0]-clip.clip_scale[1])+clip.clip_mean
    if clip.normalize == "standardized":
        Li_in_plot = Li_in_plot*clip.Li_std+clip.Li_mean
        C_in_plot = C_in_plot*clip.C_std+clip.C_mean
        Li_out_plot = Li_out_plot*clip.Li_std+clip.Li_mean
        C_out_plot = C_out_plot*clip.C_std+clip.C_mean
        batch_sat_plot = batch_sat_plot*clip.clip_std+clip.clip_mean

    clip.clip_to_obs(Li_out_plot, C_out_plot, r_at = clip.r_sat, clip_at_sat = clip_at_sat, batch_labels = batch_labels)
    sat_plot = clip.B_clip_pred[:,0]
    
    if clip_at_sat == True:
        sat_plot_pred_Li = clip.B_clip_pred_Li
        sat_plot_pred_C = clip.B_clip_pred_C
        sat_plot_label_Li = clip.B_clip_label_Li
        sat_plot_label_C = clip.B_clip_label_C

        sat_Li_residuals = np.ravel(sat_plot_label_Li)-np.ravel(sat_plot_pred_Li)
        sat_C_residuals = np.ravel(sat_plot_label_C)-np.ravel(sat_plot_pred_C)

        RMSE_sat_Li = np.sqrt(np.mean(sat_Li_residuals**2))
        RMSE_sat_C = np.sqrt(np.mean(sat_C_residuals**2))

        print("RMSE sat_height_Li: ", RMSE_sat_Li)
        print("RMSE sat_height_C: ", RMSE_sat_C)

    # Residuals
    Li_residuals = np.ravel(Li_in_plot)-np.ravel(Li_out_plot)
    C_residuals = np.ravel(C_in_plot)-np.ravel(C_out_plot)
    sat_residuals = np.ravel(batch_sat_plot)-sat_plot

    # MSE error of plotted output and label
    RMSE_Li = np.sqrt(np.mean(Li_residuals**2))
    RMSE_C = np.sqrt(np.mean(C_residuals**2))
    RMSE_sat = np.sqrt(np.mean(sat_residuals**2))

    print("RMSE sat: ", RMSE_sat)
    print("RMSE Li: ", RMSE_Li)
    print("RMSE C: ", RMSE_C)

    fig = plt.figure(figsize=figsize, constrained_layout=True) # Initiate figure with constrained layout

    if clip_at_sat == True:
        gs = fig.add_gridspec(3, 2) # Add 3x2 grid
        
        ax4 = fig.add_subplot(gs[2, 0]) 
        ax4.set_title('Lithosphere residuals at satellite altitude')
        ax4.set_xlabel('sat_Li_in - sat_Li_out')
        ax4.set_ylabel('Count')
        im4 = ax4.hist(sat_Li_residuals, bins = bins)      

        ax5 = fig.add_subplot(gs[2, 1]) 
        ax5.set_title('Core residuals at satellite altitude')
        ax5.set_xlabel('sat_C_in - sat_C_out')
        ax5.set_ylabel('Count')
        im5 = ax5.hist(sat_C_residuals, bins = bins)   

    else:
        gs = fig.add_gridspec(2, 2) # Add 2x2 grid

    ax1 = fig.add_subplot(gs[1, 0]) 
    ax1.set_title('Lithosphere residuals')
    #ax1.xlabel("Lithosphere residuals")
    #ax1.set_xlabel('(Li_in - Li_out)^2/Li_RMSE')
    ax1.set_xlabel('Li_in - Li_out')
    ax1.set_ylabel('Count')
    #im1 = ax1.hist(Li_residuals*np.abs(Li_residuals)/RMSE_Li, bins = bins)
    im1 = ax1.hist(Li_residuals, bins = bins)

    ax2 = fig.add_subplot(gs[1, 1]) 
    ax2.set_title('Core residuals')
    #ax2.set_xlabel('(C_in - C_out)^2/C_RMSE')
    ax2.set_xlabel('C_in - C_out')
    ax2.set_ylabel('Count')
    #im2 = ax2.hist(C_residuals*np.abs(C_residuals)/RMSE_C, bins = bins)
    im2 = ax2.hist(C_residuals, bins = bins)

    ax3 = fig.add_subplot(gs[0, :]) 
    ax3.set_title('Sat residuals')
    #ax3.set_xlabel('(Sat_in - Sat_out)^2/Sat_RMSE')
    ax3.set_xlabel('Sat_in - Sat_out')
    ax3.set_ylabel('Count')
    #im3 = ax3.hist(sat_residuals*np.abs(sat_residuals)/RMSE_sat, bins = bins)
    im3 = ax3.hist(sat_residuals, bins = bins)

    plt.show()


def plot_latent_parameters(epoch_i, mu, log_var, z):
    import numpy as np
    import matplotlib.pyplot as plt

    mu_in = mu[epoch_i].T
    var_in = np.exp(log_var[epoch_i].T)
    #, c = (mu_in-var_in)**2

    fig = plt.figure(figsize=(14,7), constrained_layout=True) # Initiate figure with constrained layout
    gs = fig.add_gridspec(1, 2) # Add 1x2 grid

    ax1 = fig.add_subplot(gs[0, 0]) 
    ax1.set_title('Latent parameters')
    #ax1.set_xlabel("Latent parameter index")
    ax1.set_xlabel("Mean")
    ax1.set_ylabel("Variance")
    ax1.grid()
    #im1 = ax1.plot(mu[epoch_i][idx_batch].T, "*", linestyle=(0, (1, 5)))
    im1 = ax1.plot(mu_in.reshape(1,-1), var_in.reshape(1,-1), 'o', markersize = 20)

    #ax2 = fig.add_subplot(gs[0, 1]) 
    #ax2.set_title('sigma')
    #ax2.set_xlabel("Latent variable")
    #ax2.set_ylabel("Variance")
    #im2 = ax2.plot(np.exp(log_var[epoch_i][idx_batch].T), "*", linestyle=(0, (1, 5)))
    #im2 = ax2.plot(np.exp(log_var[epoch_i][idx_batch].T), "*")

    ax3 = fig.add_subplot(gs[0, 1]) 
    ax3.set_title('z')
    ax3.set_ylabel("Sampled value")
    ax3.set_xlabel("Samples")
    im3 = ax3.plot(z[epoch_i], "*", linestyle=(0, (1, 5)))


def handle_poles(grid_core, setup_core, grid_sat, setup_sat):
    import numpy as np    
    
    if grid_core is not None:
        idx_end_core = grid_core["N"]-1
        grid_core["lat"] = np.delete(grid_core["lat"],[0,idx_end_core],0)
        grid_core["lon"] = np.delete(grid_core["lon"],[0,idx_end_core],0)
        grid_core["N"] = idx_end_core-1
        
        grid_core["n_regions"] = np.delete(grid_core["n_regions"],-1,1)
        grid_core["n_regions"] = np.delete(grid_core["n_regions"],0,1)
        
        grid_core["s_cap"] = np.delete(grid_core["s_cap"],-1,1)
        grid_core["s_cap"] = np.delete(grid_core["s_cap"],0,1)
        
        """
        data_core["lat"] = np.delete(data_core["lat"],[0,idx_end_core],0)
        data_core["lon"] = np.delete(data_core["lon"],[0,idx_end_core],0)
        data_core["radius"] = np.delete(data_core["radius"],[0,idx_end_core],0)
        data_core["data"] = np.delete(data_core["data"],[0,idx_end_core],0)
        data_core["N"] = idx_end_core-1
        """
        setup_core["N"] = idx_end_core-1
        
        
        
        if grid_core["sph_dist"] is not None:
            grid_core["sph_dist"] = np.delete(grid_core["sph_dist"],[0,idx_end_core],0)
            grid_core["sph_dist"] = np.delete(grid_core["sph_dist"],[0,idx_end_core],1)
    
    if grid_sat is not None:
        idx_end_sat = grid_sat["N"]-1
        grid_sat["lat"] = np.delete(grid_sat["lat"],[0,idx_end_sat],0)
        grid_sat["lon"] = np.delete(grid_sat["lon"],[0,idx_end_sat],0)
        grid_sat["N"] = idx_end_sat-1
        
        """
        data_sat["lat"] = np.delete(data_sat["lat"],[0,idx_end_sat],0)
        data_sat["lon"] = np.delete(data_sat["lon"],[0,idx_end_sat],0)
        data_sat["radius"] = np.delete(data_sat["radius"],[0,idx_end_sat],0)
        data_sat["data"] = np.delete(data_sat["data"],[0,idx_end_sat],0)
        data_sat["N"] = idx_end_sat-1
        """
        setup_sat["N"] = idx_end_sat-1
        
        if grid_sat["sph_dist"] is not None:
            grid_sat["sph_dist"] = np.delete(grid_sat["sph_dist"],[0,idx_end_sat],0)
            grid_sat["sph_dist"] = np.delete(grid_sat["sph_dist"],[0,idx_end_sat],1)
        
        #if np.logical_and(data_core is not None, grid_core is not None):
        if grid_core is not None:
            #return grid_core, data_core, setup_core, grid_sat, data_sat, setup_sat
            return grid_core, setup_core, grid_sat, setup_sat
        #elif np.logical_and(data_sat is None, grid_sat is None):            
        else:
            #return grid_sat, data_sat, setup_sat
            return grid_sat, setup_sat
    else:
        return grid_core, setup_core


def find_sort_d(grid_core, max_dist = 2000):
    import numpy as np
    range_d = grid_core["sph_dist"].ravel() < max_dist
    idx_range = np.array(np.where(range_d == True)).ravel()
    val_range = grid_core["sph_dist"].ravel()[idx_range]
    idx_sort_val_range = np.argsort(val_range)
    sort_d = idx_range[idx_sort_val_range]
    return sort_d


"""
FUNCTIONS RELATED TO GREEN'S
"""

def Gr_vec(r_s, r_d, lat_s, lat_d, lon_s, lon_d, angdist_out = False):
    # s for target surface, d for data
    import numpy as np

    theta_s, theta_d, lon_s, lon_d = map(np.radians, [np.matrix(90.0-lat_s), np.matrix(90.0-lat_d), np.matrix(lon_s), np.matrix(lon_d)])
    
    r_s = np.matrix(r_s)
    r_d = np.matrix(r_d)
    
    mu = np.cos(theta_d.T)*np.cos(theta_s)+np.multiply(np.sin(theta_d.T)
    *np.sin(theta_s),np.cos(lon_d.T-lon_s))
    
    h = r_s.T/r_d
    
    def rs(r_s,r_d, mu):
        r_d_sq = np.power(r_d,2)
        r_s_sq = np.power(r_s,2)
        rr_ds = r_d.T*r_s
        rr_ds_mu = 2*np.multiply(rr_ds,mu)
        rr_ds_sq_sum = r_d_sq.T+r_s_sq
        R = np.sqrt(rr_ds_sq_sum-rr_ds_mu)
        f = R.T/r_d
        return f
    
    f = rs(r_s,r_d, mu)

    h_sq = np.power(h,2)
    
    f_cb = np.power(f,3)
    
    G_r = (1/(4*np.pi)*np.multiply(h_sq,(1-h_sq))/f_cb).T
    if angdist_out == True:
        return G_r, mu
    else:
        return G_r


def take_along_axis(arr, ind, axis):
    import numpy as np
    """
    ... here means a "pack" of dimensions, possibly empty

    arr: array_like of shape (A..., M, B...)
        source array
    ind: array_like of shape (A..., K..., B...)
        indices to take along each 1d slice of `arr`
    axis: int
        index of the axis with dimension M

    out: array_like of shape (A..., K..., B...)
        out[a..., k..., b...] = arr[a..., inds[a..., k..., b...], b...]
    """
    if axis < 0:
       if axis >= -arr.ndim:
           axis += arr.ndim
       else:
           raise IndexError('axis out of range')
    ind_shape = (1,) * ind.ndim
    ins_ndim = ind.ndim - (arr.ndim - 1)   #inserted dimensions

    dest_dims = list(range(axis)) + [None] + list(range(axis+ins_ndim, ind.ndim))

    # could also call np.ix_ here with some dummy arguments, then throw those results away
    inds = []
    for dim, n in zip(dest_dims, arr.shape):
        if dim is None:
            inds.append(ind)
        else:
            ind_shape_dim = ind_shape[:dim] + (-1,) + ind_shape[dim+1:]
            inds.append(np.arange(n).reshape(ind_shape_dim))

    return arr[tuple(inds)]


def greens_differentials(grid):
    import numpy as np
    s_cap = grid["s_cap"].T
    s_cap_diff = np.diff(s_cap,axis=0)

        
    n_regions = grid["n_regions"].T
    #if n_regions[0] == 1:
    s_cap_diff = np.vstack((s_cap[0],s_cap_diff))    
    s_cap_diff[-1] =  s_cap_diff[0]
    d_theta_core = np.empty([0,1],dtype=float)
    d_phi_core = np.empty([0,1],dtype=float)
        
    for i in range(0,len(n_regions)):
            
        d_theta_core = np.vstack((d_theta_core,(s_cap_diff[i]*np.ones(int(n_regions[i]))).T))
            
        d_phi_core = np.vstack((d_phi_core,(2*np.pi/n_regions[i]*np.ones((int(n_regions[i]),1)))))
    
    theta_core = np.matrix(90.0-grid["lat"])*np.pi/180.0
    
    return np.multiply(np.multiply(d_theta_core,d_phi_core),np.sin(theta_core.T))


"""
FUNCTIONS TO CALCULATE EQUAL AREA SPHERICAL COORDINATES

Assumptions:
    - dim is always 2
    - N > 2
"""

#% sradius_of_cap
def sradius_of_cap(area):
    import numpy as np
    #s_cap = 2*np.emath.arcsin(np.sqrt(area/np.pi)/2)
    s_cap = 2*np.arcsin(np.sqrt(area/np.pi)/2)
    return s_cap


#% area_of_sphere
def area_of_sphere():
    import scipy.special as scis
    import numpy as np
    dim = 2
    power = (dim+1)/2
    area = (2*np.pi**power/scis.gamma(power))
    return area


#% area_of_ideal_region
def area_of_ideal_region(N):
    area = area_of_sphere()/N
    return area


#% polar_colat
def polar_colat(N):
    c_polar = sradius_of_cap(area_of_ideal_region(N))
    return c_polar


#% num_collars
def num_collars(N,c_polar,a_ideal):
    import numpy as np
    
    n_collars = np.zeros(np.size(N)).T
    #enough = np.logical_and(N > 2, a_ideal > 0)
    n_collars = max(1,np.round((np.pi-2*c_polar)/a_ideal))
    
    return n_collars


#% ideal_collar_angle
def ideal_collar_angle(N):
    dim = 2
    angle = area_of_ideal_region(N)**(1/dim)
    return angle


#% area_of_cap
def area_of_cap(s_cap):
    import numpy as np
    area = 4*np.pi*np.sin(s_cap/2)**2
    return area


#% area_of_collar
def area_of_collar(a_top, a_bot):
    area = area_of_cap(a_bot) - area_of_cap(a_top);
    return area


#% ideal_region_list
def ideal_region_list(N,c_polar,n_collars):
    import numpy as np
    r_regions = np.zeros((1,2+int(n_collars))).T
    r_regions[0] = 1
    if n_collars > 0:
        a_fitting = (np.pi-2*c_polar)/n_collars
        ideal_region_area = area_of_ideal_region(N)
        for collar_n in range(1,int(n_collars)+1):
            ideal_collar_area = area_of_collar(c_polar+(collar_n-1)*a_fitting, c_polar+collar_n*a_fitting)
            r_regions[0+collar_n] = ideal_collar_area / ideal_region_area
            
    r_regions[1+int(n_collars)] = 1
    
    return r_regions


#% round_to_naturals   
def round_to_naturals(N,r_regions):
    import numpy as np
    n_regions = r_regions
    discrepancy = 0
    for zone_n in range(0,np.size(r_regions,0)):
        n_regions[zone_n] = np.round(r_regions[zone_n]+discrepancy)
        discrepancy = discrepancy+r_regions[zone_n]-n_regions[zone_n]
    
    return n_regions


#% cap_colats
def cap_colats(N,c_polar,n_regions):
    import numpy as np
    c_caps = np.zeros(np.size(n_regions)).T
    c_caps[0] = c_polar
    ideal_region_area = area_of_ideal_region(N)
    n_collars = np.size(n_regions,0)-2
    subtotal_n_regions = 1
    for collar_n in range(1,n_collars+1):
        subtotal_n_regions = subtotal_n_regions+n_regions[0+collar_n]
        c_caps[collar_n+0] = sradius_of_cap(subtotal_n_regions*ideal_region_area)
    
    c_caps[0+n_collars+1] = np.pi
    
    return c_caps


#% eq_caps
def eq_caps(N):
    c_polar = polar_colat(N)

    n_collars = num_collars(N,c_polar,ideal_collar_angle(N))
    
    r_regions = ideal_region_list(N,c_polar,n_collars)
    
    n_regions = round_to_naturals(N,r_regions)
    
    s_cap = cap_colats(N,c_polar,n_regions)
    
    return s_cap, n_regions
    

#% circle_offset   
def circle_offset(n_top,n_bot):
    import numpy as np
    #from math import gcd
    
    offset = (1/n_bot - 1/n_top)/2 + np.gcd(n_top,n_bot)/(2*n_top*n_bot)
    return offset


#% eq_point_set_polar
def eq_point_set_polar(N):
    import numpy as np
    from math import floor
    
    s_cap, n_regions = eq_caps(N)
    
    n_collars = np.size(n_regions,0)-2
    
    points_s = np.zeros((N,2))
    point_n = 1
    offset = 0
    
    cache_size = floor(n_collars/2)
    cache = list()

    for collar_n in range(0,n_collars):
        s_top = s_cap[collar_n]
        s_bot = s_cap[collar_n+1]
        n_in_collar = n_regions[collar_n+1]
        
        twin_collar_n = n_collars-collar_n+1
        
        if (twin_collar_n <= cache_size and np.size(cache[twin_collar_n]) == n_in_collar):
            points_1 = cache[twin_collar_n]
            
        else:
            sector = np.arange(1,n_in_collar+1)
            s_cap_1 = sector*2*np.pi/n_in_collar
            #n_regions_1 = np.ones(len(sector))
            
            points_1 = s_cap_1 - np.pi/n_in_collar
            
            cache.append(points_1)
            
        s_point = (s_top+s_bot)/2
        
        point_1_n = np.arange(0,np.size(points_1))

        #print(point_n+point_1_n)
        points_s[point_n+point_1_n,0] = (points_1[point_1_n]+2*np.pi*offset)%(2*np.pi)

        offset = offset + circle_offset(int(n_in_collar),int(n_regions[2+collar_n]))
        offset = offset - floor(offset)

        points_s[point_n+point_1_n,1] = s_point
        point_n = point_n + np.size(points_1)
    
    points_s[point_n,:] = np.zeros((1,2))
    points_s[point_n,1] = np.pi
    
    return points_s