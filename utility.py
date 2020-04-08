"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Utility functions by Mikkel Otzen
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import matplotlib.pyplot as plt
import os

utility_abs_path = os.path.dirname(__file__)

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

def printProgressBar (iteration, total, *args, subject='', prefix = '', suffix = '', decimals = 1, length = 10, fill = 'O'):
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
        print('\r%s |%s| %s%% %s %s. Counter: %s/%s, Running error magnitude: %.1f' % (prefix, bar, percent, suffix, subject, iteration, total, args[0]), end = '\r')
    else:
        print('\r%s |%s| %s%% %s %s. Counter: %s/%s' % (prefix, bar, percent, suffix, subject, iteration, total), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def plot_cartopy_global(lat = None, lon = None, data=None, limits_data = None, shape = (360,720), plot_quality = None, unit = "[nT]", cmap = plt.cm.RdBu_r, projection_transformation = "Mollweide", figsize=(10,10), title='Cartopy Earth plot', lat_0 = 0.0, lon_0 = 0.0, point_size=2, showfig=True, norm_class = False, scale_uneven = False, flip_shape = False, flip_grid = True, transpose_grid = False, shift_grid = False, savefig = False, dpi = 100, path = None, saveformat = ".png"):

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
        axgr = AxesGrid(fig, 111, axes_class=axes_class,
                        nrows_ncols=(1, 1),
                        axes_pad=0.1,
                        cbar_location='bottom',
                        cbar_mode='single',
                        cbar_pad=0.05,
                        cbar_size='5%',
                        label_mode='')  # note the empty label_mode

        axgr[0].coastlines()
        axgr[0].set_global()
        
        if data is None:
            axgr[0].scatter(lon, lat, s=point_size, transform=ccrs.PlateCarree(), cmap=cmap)

        else:
            cb = axgr[0].scatter(lon, lat, s=point_size, c=data, transform=ccrs.PlateCarree(), vmin = vmin, vmax = vmax, cmap=cmap, norm = norm_in)

            axgr.cbar_axes[0].colorbar(cb)
            cax = axgr.cbar_axes[0]
            axis = cax.axis[cax.orientation]
            axis.label.set_text('%s %s' %(title,unit))
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
            data_in = np.hstack((data_in[:,shape[0]:],data_in[:,:shape[0]]))

        cs = ax.imshow(data_in,  vmin = vmin, vmax = vmax, cmap = cmap, norm=norm_in, transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])

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
        axgr = AxesGrid(fig, 111, axes_class=axes_class,
                        nrows_ncols=(1, 1),
                        axes_pad=0.1,
                        cbar_location='bottom',
                        cbar_mode='single',
                        cbar_pad=0.05,
                        cbar_size='5%',
                        label_mode='')  # note the empty label_mode

        axgr[0].coastlines()
        axgr[0].set_global()
                
        cb = axgr[0].scatter(lon, lat, s=point_size, c=limits_data, transform=ccrs.PlateCarree(), vmin = vmin, vmax = vmax, cmap=cmap, norm = norm_in)

        axgr.cbar_axes[0].colorbar(cb)
        cax = axgr.cbar_axes[0]
        axis = cax.axis[cax.orientation]
        axis.label.set_text('%s %s' %(title,unit))

        def animate(i):
            cb = axgr[0].scatter(lon, lat, s=point_size, c=data[:,i], transform=ccrs.PlateCarree(), cmap=cmap, norm = norm_in)
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

        cs = ax.imshow(data_init,  vmin = vmin, vmax = vmax, cmap = cmap, norm=norm_in, transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])

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
            cs = ax.imshow(data_i,  vmin = vmin, vmax = vmax, cmap = cmap, norm=norm_in, transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])
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

def plot_power_spectrum(p_spec, figsize=(14,8)):
    import matplotlib.pyplot as plt
    import numpy as np
    ns = np.arange(1,len(p_spec))
    n_ticks = np.append(np.append(np.array([1,]),np.arange(10,np.max(ns),step=10)),np.max(ns))
    plt.figure(figsize=figsize)
    plt.plot(ns, p_spec[1:])
    plt.yscale('log')
    plt.xlabel("Spherical harmonic degree")
    plt.ylabel("Power [nt²]")
    plt.xticks(n_ticks, fontsize="small")
    plt.grid(alpha=0.3)
    plt.show()

def plot_p_spec(g_spec, p_spec_height, nmax, model_dict = None, figsize = (14,8), lwidth = 2, step = 5, ensemble = False, label = "ensemble", color = "lightgray", legend_loc = "best", r_ref = 6371.2):

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy as sp
    import scipy.io as spio
    import pyshtools

    # ChaosMagPy modules
    from chaosmagpy import load_CHAOS_matfile
    from chaosmagpy.model_utils import synth_values
    from chaosmagpy.data_utils import mjd2000

    plt.figure(figsize=figsize)

    ns = np.arange(1,nmax+1)
    n_ticks = np.append(np.array([1, 5, 10,]),np.arange(15,np.max(ns)+step,step=step))

    if ensemble == False:
        ens_cilm = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,),g_spec))))
        p_spec = pyshtools.gravmag.mag_spectrum(ens_cilm, r_ref, p_spec_height, degrees = np.arange(1,np.shape(ens_cilm)[1])) # degrees to skip zeroth degree
        p_spec = p_spec[:nmax]
    else:
        N_ensembles = np.shape(g_spec)[-1]

        for i in np.arange(0,N_ensembles):
            ens_cilm = np.array(pyshtools.shio.SHVectorToCilm(np.hstack((np.zeros(1,),g_spec[:,i]))))
            p_spec = pyshtools.gravmag.mag_spectrum(ens_cilm, r_ref, p_spec_height, degrees = np.arange(1,np.shape(ens_cilm)[1]))
            p_spec = p_spec[:nmax]
            if i == 0:
                plt.plot(ns, p_spec[:nmax], color=color, label = label)
            else:
                plt.plot(ns, p_spec[:nmax], color=color)

    if model_dict is not None: # Load models
        WDMAM2 = spio.loadmat('lithosphere_prior/grids/models/WDMAM2.mat')
        LCS1 = load_shc("lithosphere_prior/grids/models/LCS-1.shc")
        MF7 = load_shc("lithosphere_prior/grids/models/MF7.shc")
        EMM2017 = np.loadtxt('lithosphere_prior/grids/models/EMM2017.COF',comments="%",skiprows=1)

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
        Gauss_in_pomme = np.loadtxt('lithosphere_prior/grids/models/POMME_6_main_field.txt')
        g_pomme = np.hstack((np.zeros(1,), gauss_vector(Gauss_in_pomme, 60, i_n = 2, i_m = 3)))
        cilm_pomme = pyshtools.shio.SHVectorToCilm(g_pomme)

        # CHAOS 7
        N_chaos = 20
        CHAOS7 = load_CHAOS_matfile('lithosphere_prior/grids/models/CHAOS-7.mat')
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

            plt.plot(use_ns, use_p_spec, color="C{}".format(i), label = key, linewidth = lwidth)
            i += 1

    plt.yscale('log')
    plt.xlabel("degree n")
    plt.ylabel("Power [nt²]")
    plt.xticks(n_ticks, fontsize="small")
    plt.grid(alpha=0.3)
    plt.legend(loc = legend_loc)
    plt.show()


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


def plot_CLiP_data(key, labels, shape_s, shape_Li, shape_C):
    # Easy random plot synth sat, lithosphere, core

    import matplotlib.pyplot as plt
    import matplotlib.ticker as tick

    SF = tick.ScalarFormatter() # Formatter for colorbar
    SF.set_powerlimits((3, 3)) # Set sci exponent used

    fig = plt.figure(figsize=(16,10), constrained_layout=True) # Initiate figure with constrained layout
    gs = fig.add_gridspec(2, 2) # Add 2x2 grid

    ax1 = fig.add_subplot(gs[0, :]) # Use full row
    ax1.set_title('Synth sat obs, B_r [nT]')
    im1 = ax1.imshow(ens_s_sat[:,key].reshape(shape_s), cmap = plt.cm.RdBu_r)
    fig.colorbar(im1, ax=ax1, orientation = "horizontal", shrink=0.3, format = SF)

    ax2 = fig.add_subplot(gs[1, 0]) # Use one 
    ax2.set_title('Lithosphere, B_r [nT]')
    im2 = ax2.imshow(ens_Li[:,labels[key][0]].reshape(shape_Li).T, cmap = plt.cm.RdBu_r)
    fig.colorbar(im2, ax=ax2, orientation = "horizontal", shrink=0.6)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('Core, B_r [nT]')
    im3 = ax3.imshow(ens_C[:,labels[key][1]].reshape(shape_C).T, cmap = plt.cm.RdBu_r)
    fig.colorbar(im3, ax=ax3, orientation = "horizontal", shrink=0.6, format = SF)
    plt.show()


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

def gauss_vector(g_in, N_deg, i_n = 0, i_m = 1):
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