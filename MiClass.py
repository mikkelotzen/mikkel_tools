#!/usr/bin/env python3

# Mikkel Otzen modules
import numpy as np
import scipy as sp
#from utility import printProgressBar
import scipy.interpolate as itp
import GMT_tools as gt
import time
import pyshtools
import mikkel_tools.utility as mt_util

# Simon Williams modules
#import os
#from scipy.interpolate import interp1d
#from scipy.interpolate import RegularGridInterpolator 
#from scipy.optimize import minimize_scalar
#from pyresample import image, geometry
#import xarray as xr
#import VIM_tools as vt

# ChaosMagPy modules
from chaosmagpy import load_CHAOS_matfile
from chaosmagpy.model_utils import synth_values
from chaosmagpy.data_utils import mjd2000

class MiClass(object):    
    """ Class for generating ensemble of Lithospheric priors """
  
    def __init__(self, sat_height = 350):
        
        self.a = 6371.2
        self.r_sat = self.a + sat_height
        self.r_cmb = 3480.0
        self.rad = np.pi/180
        self.mu0 = 4*np.pi*10**(-7)   

        self.attribute_string = {"string_return":"", "string_return_previous":"", "keys":list(), "keys_old":list()}
        
    def __repr__(self):
        return "Class: m_class()"
    
    
    def __str__(self):
        self.attribute_string_update()
        return self.attribute_string["string_return"]


    def attribute_string_update(self):
        
        self.attribute_string["string_return_previous"] = self.attribute_string["string_return"]
        self.attribute_string["keys_old"] = self.attribute_string["keys"].copy()
        self.attribute_string["keys"] = list()
        
        keys = self.__dict__.keys() # Keys of self dict
        
        key_width = max(len(key) for key in keys) + 2 # Longest key + padding
        
        string_return = "\n______current attributes______\n\n" # Explanatory title of print
        
        # Add to string_return each attribute and its value or shape in the self dict.
        # Printing is left justified to the longest key to achieve equal column print.
        for entry in keys: 
            
            self.attribute_string["keys"].append(entry)
            
            dict_obj = self.__dict__[entry]
            
            if type(dict_obj) is not dict: # avoid printing dicts and lists
            
                dict_obj_shape = np.shape(dict_obj)
                dict_obj_shape_len = len(dict_obj_shape)
                
                if dict_obj_shape == ():
                    string_return = string_return + (str(entry) + ": ").ljust(key_width) + str(dict_obj) + "\n"
                elif dict_obj_shape_len >= 3:
                    string_return = string_return + (str(entry) + ": ").ljust(key_width) + str(dict_obj_shape) + "\n"
                elif dict_obj_shape_len == 2 and dict_obj_shape[1] > 1:
                    string_return = string_return + (str(entry) + ": ").ljust(key_width) + str(dict_obj_shape) + "\n" 
                elif dict_obj_shape_len == 1 or dict_obj_shape[1] == 1:
                    string_return = string_return + (str(entry) + ": ").ljust(key_width) + str(dict_obj_shape) + ", Max/Min: " + "{:.2f}".format(np.max(dict_obj)) + " / " + "{:.2f}".format(np.min(dict_obj)) + "\n"
                else:
                    string_return = string_return + (str(entry) + ": ").ljust(key_width) + str(dict_obj_shape) + ", Max/Min: [" + (("{:.2f}, "*dict_obj_shape[1]).format(*np.max(dict_obj, axis=0)) + "]").replace(", ]","]") + " / [" + (("{:.2f}, "*dict_obj_shape[1]).format(*np.min(dict_obj, axis=0)) + "]").replace(", ]","]") + "\n"

        self.attribute_string["string_return"] = string_return
        
       
    def attribute_string_update_print(self):
        
        self.attribute_string_update()
        
        # If all old keys are not in new keys, some attributes have been deleted.
        # In this case, print all again.
        if all([i in self.__dict__ for i in self.attribute_string["keys_old"]]) != True:
            print("\n______Some attributes have been deleted, printing all______")
            print(self.attribute_string["string_return"])
        else:
            print("\n______new attributes______\n")
            print(self.attribute_string["string_return"].split("\n", self.attribute_string["string_return_previous"].count("\n"))[-1])      
    

    def grid_even_spaced(self, grid_size=0.25, r_at = None, wgs84=False, square_grid = False):
        """ Hemant & Masterton style evenly spaced grid generation """

        if r_at is None:
            r_at = self.a

        if square_grid == True:
            lat, lon = np.meshgrid(np.arange(-90+grid_size/2, 90, grid_size),np.arange(-180+grid_size/2, 180, 2*grid_size))
        else:
            lat, lon = np.meshgrid(np.arange(-90+grid_size/2, 90, grid_size),np.arange(-180+grid_size/2, 180, grid_size))
        
        self.grid_theta_len = len(lat.T)
        self.grid_phi_len = len(lon)
        self.grid_shape = (self.grid_phi_len, self.grid_theta_len)

        self.grid_theta = 90 - np.flip(np.ravel(lat.T)).reshape(-1,)
        self.grid_phi = np.ravel(lon.T).reshape(-1,) + 180
        
        # Possible use of WGS84 radius
        if wgs84 == False:
            self.grid_radial = np.ones((len(self.grid_theta),))*r_at
        else:
            self.wgs84_a = 6378.137
            self.wgs84_c = 6356.752
            self.wgs84_f = (self.wgs84_a-self.wgs84_c)/self.wgs84_a
            self.grid_radial = self.wgs84_a*(1-self.wgs84_f*np.sin(self.grid_theta)**2)
    
    
    def grid_glq(self, nmax = 14, r_at = None):

        if r_at is None:
            r_at = self.a

        # Gauss-Legendre Quadrature Grid
        
        lat_glq, lon_glq = pyshtools.expand.GLQGridCoord(nmax,extend=False)
        #lon_glq = np.round(lon_glq[:-1])

        #lon_glq = np.arange(0,2*np.pi,np.pi/nmax)*180/np.pi

        self.grid_zero, grid_w = pyshtools.expand.SHGLQ(nmax)
        self.grid_w_shtools = grid_w.copy()
        
        grid_theta_len = len(lat_glq)
        grid_phi_len = len(lon_glq)
        self.grid_shape = (grid_phi_len, grid_theta_len)

        #grid_w = np.polynomial.legendre.legweight(np.flipud(np.cos((90-lat_glq)*self.rad)))

        weights, none = np.meshgrid(grid_w,lon_glq,indexing='ij') # Get weights for quadrature on grid
        self.grid_w = np.ravel(weights)
        
        #lat_glq, lon_glq = np.meshgrid(lat_glq,lon_glq)
        lon_glq, lat_glq = np.meshgrid(lon_glq,lat_glq)


        self.grid_radial  = np.ones((len(lat_glq.ravel()),))*r_at
        self.grid_theta = 90 - np.ravel(lat_glq).reshape(-1,)
        self.grid_phi = np.ravel(lon_glq).reshape(-1,)
        self.grid_N = np.shape(self.grid_radial)[0]

        self.grid_nmax = nmax


    def grid_glq_np(self, nmax = 14, r_at = None):

        gauss_leg = np.polynomial.legendre.leggauss(nmax) # Use built-in numpy function to generate grid
        
        # Set lat and lon range from estimated grid
        theta = np.flipud(np.arccos(gauss_leg[0]).reshape(-1,1))*180/np.pi
        phi = np.arange(0,2*np.pi-np.pi/nmax,np.pi/nmax)*180/np.pi
        
        weights, none = np.meshgrid(gauss_leg[1],phi,indexing='ij') # Get weights for quadrature on grid
        self.grid_w = np.ravel(weights)
        
        # Compute full lat/lon grid
        theta, phi = np.meshgrid(theta, phi,indexing='ij')

        self.grid_radial = np.ones((len(theta),))*r_at
        self.grid_theta = theta.ravel()
        self.grid_phi = phi.ravel()
        self.grid_N = 2*nmax**2
        self.grid_nmax = nmax


    def grid_equal_area(self, N_grid = 1000, r_at = None, poles_remove = False):

        N_grid_orig = N_grid
        check_flag = False

        while check_flag is False:
            points_polar = mt_util.eq_point_set_polar(N_grid) # Compute grid with equal area grid functions
            
            # Set lat and lon from estimated grid
            lon = points_polar[:,0]*180/np.pi
            lat = 90 - points_polar[:,1]*180/np.pi

            # Determine equal area grid specifics used for defining the integration area
            s_cap, n_regions = mt_util.eq_caps(N_grid)
            self.n_regions = n_regions.T
            self.s_cap = s_cap
            
            if N_grid == int(np.sum(n_regions)):
                check_flag = True
                if N_grid_orig - N_grid != 0:
                    print("")
                    print("___ CHANGES TO GRID ___")
                    print("N = {}, not compatible for equal area grid".format(N_grid_orig))
                    print("N has been set to {}".format(N_grid))
                
            else:
                N_grid -= 1
        
        if poles_remove == True:
            # Remove the first and last grid points (the poles) and the corresponding structure related components
            idx_end_core = N_grid-1
            lat = np.delete(lat,[0,idx_end_core],0)
            lon = np.delete(lon,[0,idx_end_core],0)
            N_grid = idx_end_core-1
            
            self.n_regions = np.delete(self.n_regions,-1,1)
            self.n_regions = np.delete(self.n_regions,0,1)
            
            self.s_cap = np.delete(self.s_cap,-1,0)
            self.s_cap = np.delete(self.s_cap,0,0)
            
            N_grid = idx_end_core-1

        #self.grid_theta_len = len(np.unique(lat))
        #self.grid_phi_len = len(np.unique(lon))

        self.grid_radial = np.ones((len(lat.ravel()),))*r_at
        self.grid_theta = 90 - lat.reshape(-1,)
        self.grid_phi = lon.reshape(-1,)
        self.grid_N = N_grid        


    def ensemble_random_field_init(self):
        n = 720
        h = 1/n
        
        # Generate diagonal matrix for smoooth curvature prior (2nd order smoothness)
        H = np.hstack((np.ones((n-1,1)), -2*np.ones((n-1,1)), np.ones((n-1,1)))).T
        L1 = n**2*sp.sparse.spdiags(H,[-1,0,1],n-1,n-1)#.todense()
        
        # 2D smooth curvature prior
        D1 = sp.sparse.kron(sp.sparse.eye(n-1), L1)
        D2 = sp.sparse.kron(L1, sp.sparse.eye(n-1))
        self.L = D1 + D2
        
        self.n_rf = n
        self.rf_init_flag = True

        #self.rf_lbound = 0.5
        #self.rf_lbound_l = 0.1
        #self.rf_lbound_u = 0.5       
        
   
    def ensemble_random_field_grid(self, grid, rescale_to_grid = True):
        
        self.WM_2D = self.L - 1/(self.rf_corr)**2*sp.sparse.eye((self.n_rf-1)**2); # Wittle-Matern

        X_map = np.zeros((360,720))
        
        W = np.random.randn((self.n_rf-1)**2, 1)
        X = (self.rf_gamma*(sp.sparse.linalg.spsolve(self.WM_2D,W))).reshape(self.n_rf-1,self.n_rf-1) # Correlation length specified by Wittle-Matern

        X_full = np.zeros((self.n_rf,self.n_rf))
        X_full[1:self.n_rf, 1:self.n_rf] = X
        X_map = np.resize(X_full,(360,720))
        
        if self.rf_method == "sum":
            X_map_norm = self.rf_coeff*X_map
            self.grid_rf = grid+X_map_norm

        elif self.rf_method == "replace":

            min_grid = np.min(grid)
            mean_grid = np.mean(grid)
            #X_map_norm = self.rf_coeff*X_map
            X_map_norm = X_map
            X_map_norm = np.ravel(self.rf_coeff*(X_map_norm - np.min(X_map_norm))/(np.max(X_map_norm)-np.min(X_map_norm)))
            grid_ravel = np.ravel(grid)

            idx_rf_suppress = X_map_norm<self.rf_sbound
            idx_rf_lbound = X_map_norm>self.rf_lbound
            len_suppress = len(X_map_norm[idx_rf_suppress])
            len_zero = len(X_map_norm[idx_rf_lbound])
            grid_ravel[idx_rf_lbound]=np.multiply(grid_ravel[idx_rf_lbound],X_map_norm[idx_rf_lbound])
            #X_map_norm[idx_rf_suppress]=np.random.uniform(self.rf_sbound_l,self.rf_sbound_u,len_suppress)
            grid_ravel[idx_rf_suppress]=np.random.normal(loc=np.mean(grid), scale = 0.01, size=len_suppress)
            #X_map_norm = self.rf_coeff*(X_map - np.min(X_map))/(np.max(X_map)-np.min(X_map))
            #self.grid_rf = np.multiply(grid,X_map_norm.reshape((360,720)))
            self.grid_rf = grid_ravel.reshape((360,720))

            X_map_norm_new = np.ones(np.shape(X_map_norm))
            X_map_norm_new[idx_rf_lbound] = X_map_norm[idx_rf_lbound]
            X_map_norm_new[idx_rf_suppress] = X_map_norm[idx_rf_suppress]
            X_map_norm = X_map_norm_new

        else:
            min_grid = np.min(grid)
            mean_grid = np.mean(grid)
            #X_map_norm = self.rf_coeff*X_map
            X_map_norm = X_map
            X_map_norm = np.ravel(self.rf_coeff*(X_map_norm - np.min(X_map_norm))/(np.max(X_map_norm)-np.min(X_map_norm)))

            idx_rf_suppress = X_map_norm<self.rf_sbound
            idx_rf_lbound = X_map_norm<self.rf_lbound
            len_suppress = len(X_map_norm[idx_rf_suppress])
            len_zero = len(X_map_norm[idx_rf_lbound])
            X_map_norm[idx_rf_lbound]=np.random.uniform(self.rf_lbound_l,self.rf_lbound_u,len_zero)
            #X_map_norm[idx_rf_suppress]=np.random.uniform(self.rf_sbound_l,self.rf_sbound_u,len_suppress)
            X_map_norm[idx_rf_suppress]=np.mean(grid)
            #X_map_norm = self.rf_coeff*(X_map - np.min(X_map))/(np.max(X_map)-np.min(X_map))
            self.grid_rf = np.multiply(grid,X_map_norm.reshape((360,720)))
            
        if rescale_to_grid == False:
            max_grid = np.max(grid)
            self.grid_rf = max_grid*(self.grid_rf - np.min(self.grid_rf))/(np.max(self.grid_rf)-np.min(self.grid_rf))
        
        self.grid_rf_base = X_map_norm.reshape((360,720))


    def ensemble_random_field_grid_direct(self, grid, rescale_to_grid = True):
        
        self.WM_2D = self.L - 1/(self.rf_corr)**2*sp.sparse.eye((self.n_rf-1)**2); # Wittle-Matern

        X_map = np.zeros((360,720))
        
        W_hm = np.ravel(np.vstack((grid[1:,:],grid[:,:]))[:,1:]).reshape(-1,1)

        W = np.multiply(np.random.randn((self.n_rf-1)**2, 1),W_hm)
        X = (self.rf_gamma*(sp.sparse.linalg.spsolve(self.WM_2D,W))).reshape(self.n_rf-1,self.n_rf-1) # Correlation length specified by Wittle-Matern

        X_full = np.zeros((self.n_rf,self.n_rf))
        X_full[1:self.n_rf, 1:self.n_rf] = X
        X_map = np.resize(X_full,(360,720))

        min_grid = np.min(grid)
        mean_grid = np.mean(grid)
        X_map_norm = X_map
        X_map_norm = np.ravel(self.rf_coeff*(X_map_norm - np.min(X_map_norm))/(np.max(X_map_norm)-np.min(X_map_norm)))
        grid_ravel = np.ravel(grid)

        idx_rf_suppress = X_map_norm<self.rf_sbound
        idx_rf_lbound = X_map_norm>self.rf_lbound
        len_suppress = len(X_map_norm[idx_rf_suppress])
        len_zero = len(X_map_norm[idx_rf_lbound])
        grid_ravel[idx_rf_lbound]=np.multiply(grid_ravel[idx_rf_lbound],X_map_norm[idx_rf_lbound])

        grid_ravel[idx_rf_suppress]=np.random.normal(loc=np.random.uniform(np.mean(grid)*0.5, np.mean(grid)*1.5,1), scale = 0.01, size=len_suppress)
        #grid_ravel[idx_rf_suppress]=np.random.normal(loc=np.mean(grid), scale = 0.01, size=len_suppress)
        
        self.grid_rf = grid_ravel.reshape((360,720))

        X_map_norm_new = np.ones(np.shape(X_map_norm))
        X_map_norm_new[idx_rf_lbound] = X_map_norm[idx_rf_lbound]
        X_map_norm_new[idx_rf_suppress] = X_map_norm[idx_rf_suppress]
        X_map_norm = X_map_norm_new

        if rescale_to_grid == False:
            max_grid = np.max(grid)
            self.grid_rf = max_grid*(self.grid_rf - np.min(self.grid_rf))/(np.max(self.grid_rf)-np.min(self.grid_rf))
        
        self.grid_rf_base = X_map_norm.reshape((360,720)) 


    def ensemble_sampled_field_grid(self, grid, rescale_to_grid = True):
        
        X_map = np.zeros((360,720))
        
        #W = np.zeros((360,720))
        W_grid = grid.copy()
        len_grid = len(np.ravel(grid))
        idx_grid = np.random.randint(0,len_grid, int(len_grid/2))
        np.ravel(W_grid)[idx_grid] = np.random.randn(len(idx_grid),)
        
        W = np.ravel(np.hstack((W_grid, np.random.randn(360*720).reshape(360,720)))).reshape(-1,1)
        
        X = (self.rf_gamma*(sp.sparse.linalg.spsolve(self.WM_2D,W))).reshape(self.n_rf,self.n_rf) # Correlation length specified by Wittle-Matern

        X_full = np.zeros((self.n_rf,self.n_rf))
        X_full[:self.n_rf, :self.n_rf] = X
        X_map = np.resize(X_full,(360,720))
        
        self.grid_sf = X_map
              
            
    def ensemble_B_field(self, fields = "all", N_mf = 15, g_use = None, only_nmf = False, r_at = None, define_grid = False):
        
        if g_use is None:
            g_use = self.g_vh0_ensemble

        if r_at is None:
            r_at = self.a

        if define_grid == False:
            colat_u = np.unique(np.ravel(self.grid_theta))
            lon_u = np.unique(np.ravel(self.grid_phi))
        else:
            self.grid_spaced(grid_size=define_grid)
            colat_u = np.unique(np.ravel(self.grid_theta))
            lon_u = np.unique(np.ravel(self.grid_phi))
            
        B_ensemble = list()
        B_ensemble_nmf = list()
        
        if fields == "all":
            ensemble_count = np.shape(g_use)[1]
        else:
            ensemble_count = fields
            
        for ensemble_idx in np.arange(0,ensemble_count):
            
            g_vh0 = g_use[:,ensemble_idx]

            if only_nmf == False:
                B_vh0_r, B_vh0_theta, B_vh0_phi = gt.synth_grid(g_vh0, r_at/self.a, colat_u*self.rad, lon_u*self.rad)
                B_ensemble.append(np.array([np.ravel(B_vh0_r), np.ravel(B_vh0_theta), np.ravel(B_vh0_phi)]))

            g_vh0_nmf = g_vh0.copy()
            g_vh0_nmf[:int(2*np.sum(np.arange(1,N_mf+1)+1)-N_mf)] = 0
            B_vh0_r_nmf, B_vh0_theta_nmf, B_vh0_phi_nmf = gt.synth_grid(g_vh0_nmf, r_at/self.a, colat_u*self.rad, lon_u*self.rad)
            B_ensemble_nmf.append(np.array([np.ravel(B_vh0_r_nmf), np.ravel(B_vh0_theta_nmf), np.ravel(B_vh0_phi_nmf)]))
        
        if only_nmf == False:
            self.B_ensemble = np.array(B_ensemble).T

        self.B_ensemble_nmf = np.array(B_ensemble_nmf).T


    def ensemble_B(self, g_use, nmax = 30, N_mf = 15, mf = True, nmf = False, r_at = None, grid_type = "glq"):

        if r_at is None:
            r_at = self.a


        grid_radial = self.grid_radial
        grid_theta = self.grid_theta
        grid_phi = self.grid_phi

        """
        if grid_type == "glq":
            grid_radial = self.grid_glq_radial
            grid_theta = self.grid_glq_theta
            grid_phi = self.grid_glq_phi
        elif grid_type == "even":
            grid_radial = self.grid_even_radial
            grid_theta = self.grid_even_theta
            grid_phi = self.grid_even_phi
        elif grid_type == "eqa":        
            grid_radial = self.grid_eqa_radial
            grid_theta = self.grid_eqa_theta
            grid_phi = self.grid_eqa_phi
        elif grid_type == "swarm":
            grid_radial = self.swarm_radius
            grid_theta = self.swarm_theta
            grid_phi = self.swarm_phi
        """

        # Generate design matrix for grid
        A_r, A_theta, A_phi = gt.design_SHA(r_at/self.a, grid_theta*self.rad, grid_phi*self.rad, nmax)

        if mf == True:
            B_r = np.matmul(A_r,g_use)
            B_theta = np.matmul(A_theta,g_use)
            B_phi = np.matmul(A_phi,g_use)            
            self.B_ensemble = np.stack((B_r, B_theta, B_phi), axis = 1)

            """
            if grid_type == "glq":
                self.B_ensemble_glq = B_ensemble.copy()
            elif grid_type == "even":
                self.B_ensemble_even = B_ensemble.copy()
            elif grid_type == "eqa":
                self.B_ensemble_eqa = B_ensemble.copy()
            elif grid_type == "swarm":
                self.B_ensemble_swarm = B_ensemble.copy()
            """

        if nmf == True:
            g_use_nmf = g_use.copy()
            g_use_nmf[:int(2*np.sum(np.arange(1,N_mf+1)+1)-N_mf)] = 0

            B_r_nmf = np.matmul(A_r,g_use_nmf)
            B_theta_nmf = np.matmul(A_theta,g_use_nmf)
            B_phi_nmf = np.matmul(A_phi,g_use_nmf)
            
            self.B_ensemble_nmf = np.stack((B_r_nmf, B_theta_nmf, B_phi_nmf), axis = 1)

            """
            if grid_type == "glq":
                self.B_ensemble_nmf_glq = B_ensemble_nmf.copy()
            elif grid_type == "even":
                self.B_ensemble_nmf_even = B_ensemble_nmf.copy()
            elif grid_type == "eqa":
                self.B_ensemble_nmf_eqa = B_ensemble_nmf.copy()
            elif grid_type == "swarm":
                self.B_ensemble_nmf_swarm = B_ensemble_nmf.copy()
            """

    def interpolate_grid(self, grid_in_theta, grid_out_theta, grid_in_phi, grid_out_phi, grid_in, method_int = "nearest", output = "return", save_path = ""):
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
    
            
    def power_spec(self, m_sum_cilm, r_eval):

        self.p_spec = pyshtools.gravmag.mag_spectrum(m_sum_cilm, self.a, r_eval, degrees = np.arange(1,np.shape(m_sum_cilm)[-1])) # degrees to skip zeroth degree
        
