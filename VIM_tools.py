import os
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
#import inpaint

try:
    import pygplates
    import pttx.points_spatial_tree as points_spatial_tree
    import pttx.points_in_polygons as points_in_polygons
except:
    print('Unable to load pygplates-based functions')

try:
    from pyresample import image, geometry
except: 
    print('Unable to load pyresample')
    
try:
    import healpy as hp
except:
    print('Unable to import healpy')
    
    
DEFAULT_TIMESCALE_FILE = 'Cande_Kent96_Channell95_Nakanishi_Hounslow.txt'


def glm2pot(filename,Altitude,HarmMax=120):
    
    data = np.genfromtxt(filename)

    HarmMin = 1
    #HarmMax = 185
    ind = np.logical_and(data[:,0]>=HarmMin,data[:,0]<=HarmMax)
    data=data[ind,:]

    # rearrange columns
    A=[]
    b=0
    for c in np.arange(0,data.shape[0]):

        A1=data[c,2]      
        A.append(A1)

        # Exclude h terms with order 0 from output array 
        if np.logical_not(data[c,1]==0):
            A2=data[c,3]
            A.append(A2)

    final = np.reshape(A,(np.divide(len(A),5).astype(int),5))

    f = open('coeff_reform.txt','w')
    f.write("gridfile: from matlab vh0 run\n")
    np.savetxt(f, [HarmMax], delimiter='',fmt='%i')
    np.savetxt(f, final, delimiter='',fmt='%13.5f %13.5f %13.5f %13.5f %13.5f')
    f.close()

    f = open('crustpot2z.par','w')
    f.write("coeff_reform.txt\ncrustpot2z.out\n14\n")
    f.close()

    if HarmMax==185:
        os.system('../bin/crustpot2z_%dkm_185_3comp_new < crustpot2z.par' % Altitude)
    else:
        os.system('../bin/crustpot2z_%dkm_3comp_new < crustpot2z.par' % Altitude)
    
    data = np.genfromtxt('crustpot2z.out')
    
    Xg=np.flipud(np.reshape(data[:,0],(256,512)))
    Yg=np.flipud(np.reshape(data[:,1],(256,512)))
    ModelBx=np.flipud(np.reshape(data[:,2],(256,512)))
    ModelBy=np.flipud(np.reshape(data[:,3],(256,512)))
    ModelBz=np.flipud(np.reshape(data[:,4],(256,512)))
    ModelBh=np.flipud(np.reshape(data[:,5],(256,512)))
    ModelBf=np.flipud(np.reshape(data[:,6],(256,512)))

    return ModelBx,ModelBy,ModelBz,ModelBh,ModelBf,Xg,Yg
    

def write_vh0_input(gridX,gridY,Mr,Mtheta,Mphi,filename='total_py.in'):

    dims = Mr.shape
    
    gridXX,gridYY = np.meshgrid(gridX,gridY)
    
    # this is probably unnecessary, but to stick with convention used in previous
    # studies we set the start of the file to be the north pole (colatitude=0)
    if gridY[0]>90:
        gridYY = np.flipud(gridYY)
        Mr = np.flipud(Mr)
        Mtheta = np.flipud(Mtheta)
        Mphi = np.flipud(Mphi)
    
    OutArray = np.vstack((gridYY.flatten(),\
                      gridXX.flatten(),\
                      Mr.flatten(),\
                      Mtheta.flatten(),\
                      Mphi.flatten()))

    f = open(filename,'w')
    f.write("lat,lon,mr,mtheta,mphi   0.25 degree spacing\n")
    f.write("    %d  %d\n" % (dims[0],dims[1]))
    np.savetxt(f, OutArray.T, delimiter='',fmt='%10.3f%10.3f%16.7f%16.7f%16.7f')
    f.write("0\n")
    f.close()
    


def LithosphericCooling(t,z,Model):

    numtaylor = 10
    [tg,zg] = np.meshgrid(t,z)

    if Model=='MK78':
        # Constants McKenzie 1978
        tau = 65.        # lithosphere cooling thermal decay constant
        a = 125.         # lithosphere thickness
        Tm = 1300.       # Base lithosphere temperature
        beta = 1e99     # stretching factor, infinity for oceanic crust
        G = 6.67e-11    # Gravitational constant
        alpha = 3.28e-5 # coefficient of thermal expansion
        rho = 3300.      # lithosphere density

        # McKenzie, 1978
        ConstantTerm1 = 2/np.pi;
        NthSum = np.zeros(np.size(tg))

        for n in np.arange(1,numtaylor):
            NthTerm =  (((-1)**(n+1))/n) * ((beta/(n*np.pi))*np.sin((n*np.pi)/beta))\
                * np.exp((((-n**2))*tg)/tau) * np.sin((n*np.pi*(a-zg))/a)
            NthSum = NthSum + NthTerm
        
        Tratio = 1 - (a-zg)/a + ConstantTerm1*NthSum
        Tz = Tratio*Tm

    else:
        if Model=='P&S':
            # Constants P&S
            tau = 65.         # lithosphere cooling thermal decay constant
            a = 95.           # lithosphere thickness
            Tm = 1450.        # Base lithosphere temperature
            beta = 1e99      # stretching factor, infinity for oceanic crust
            G = 6.67e-11     # Gravitational constant
            alpha = 3.138e-5 # coefficient of thermal expansion
            rho = 3330.       # lithosphere density

        elif Model=='GDH1':
            # Constants GDH1
            a = 95000.     # lithosphere thickness
            Tm = 1450.     # Base lithosphere temperature
            rho = 3300.    # lithosphere density
            k=3.138
            Cp = 1.171e3
            kappa = k/(rho*Cp) # lithosphere cooling thermal decay constant
            zg=zg*1000.
            tg=tg*1e6*31536000.
            za=170000.
            v=1/31536000
        

        ConstantTerm1 = ((zg)/a)
        NthSum = np.zeros(np.shape(tg))

        for n in np.arange(1,numtaylor):

            NthTerm =  (2/(n*np.pi)) * np.sin((n*np.pi*(zg))/a)\
                * np.exp(-((n**2)*(np.pi**2)*kappa*tg)/(a**2))

            NthSum = NthSum + NthTerm
        
        Tz = ConstantTerm1 + NthSum
        Tz = Tz*Tm

    return Tz 
    

# Define a function that will calculate components of remnant magnetisation 
# as a function of seafloor age
def MagnetisationModel(t,Mtrm,Mcrm,P,lmbda,MagMax,timescalefile=DEFAULT_TIMESCALE_FILE):

    delta_ta = np.abs(t[1]-t[0])

    # Load the timescale, and turn into a 'square wave'
    TS = np.loadtxt(timescalefile,usecols=(0,))
    TSr = np.zeros(TS.shape[0]*2)
    TSr[::2] = TS
    TSr[1:-1:2] = TS[1::]+0.00001
    TSr[-1]=600
    TSp = np.ones(TSr.shape)
    TSp[::4] = -1
    TSp[1::4] = -1

    # Create regularly sampled time and polarity arrays
    tpInterpolator = interp1d(TSr,TSp,kind='nearest',bounds_error=False,fill_value=0)
    tp = tpInterpolator(t)

    # TRM is purely a function of age
    TRM = ( Mtrm*tp*(1+P*np.exp(-t/lmbda)) )

    # CRM is a function of the sequence of reversals that follows initial crust
    # formation, hence a loop is required to get value for each point in profile
    CRM = np.zeros(t.shape)
    for j in np.arange(1,t.shape[0]):
        CrmTerm = tp[0:j]* (np.exp(-((t[j]-t[0:j])/lmbda)))*delta_ta
        CRM[j] =  np.sum(CrmTerm) * (Mcrm/lmbda)

    # This scaling is to force the magnetization at MOR to be some value (0.5 A/m)
    ScalingFactor = np.abs(TRM[0])
    TRM = (TRM/ScalingFactor)*MagMax
    CRM = (CRM/ScalingFactor)*MagMax

    return TRM,CRM


def MagnetisationCrossSection(z,Age,TimeToCurieByDepth,TRM,CRM):
    
    # By combining the magnetisation-age function with GDH1, we can get a vertical
    # cross-section model of magnetisation that reflects the time taken for
    # different depths to pass through the Curie temperature.
    OceanCrossSection = np.zeros((z.shape[0],Age.shape[0]))
    TRMg = np.zeros((z.shape[0],Age.shape[0]))
    CRMg = np.zeros((z.shape[0],Age.shape[0]))
    for i in np.arange(0,OceanCrossSection.shape[0]):
        OceanCrossSection[i,:] = Age-TimeToCurieByDepth[i]

    TRMInterpolator = interp1d(Age,TRM,kind='nearest',\
                               bounds_error=False,fill_value=0)
    TRMg = TRMInterpolator(OceanCrossSection)
    CRMInterpolator = interp1d(Age,CRM,kind='nearest',\
                               bounds_error=False,fill_value=0)
    CRMg = CRMInterpolator(OceanCrossSection)

    TRMg[OceanCrossSection<=0] = 0
    CRMg[OceanCrossSection<=0] = 0
    TRMg[np.isnan(TRMg)] = 0
    CRMg[np.isnan(CRMg)] = 0
    
    return TRMg,CRMg


def StratifiedVIM(z,TRMg,LayerDepths,LayerWeights,
                  return_cross_section=False):
    
    LayerDepths = np.hstack((0,np.array(LayerDepths).flatten()))
    dz = z[1]-z[0]
    
    TRMg_layered = np.zeros(TRMg.shape)

    for LayerTop,LayerBottom,LayerWeight in zip(LayerDepths[:-1],LayerDepths[1:],LayerWeights):
        ind = np.where((z>=LayerTop) & (z<LayerBottom))
        TRMg_layered[ind] = TRMg[ind] * LayerWeight
     
    if return_cross_section:
        return TRMg_layered*dz
    
    else:
        # do the vertical integration
        VIM = np.sum(TRMg_layered,0)*dz
        return VIM
    
    

def CurieDepthModel(Age,z,CurieTemp=580,AgeModel='GDH1'):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Based of GDH1, get a model for the thermal structure of the oceanic
    # lithosphere as a function of age
    # Note that the inputs and outputs are assumed to be in meters,
    # but are converted to kms for internal calculations
    Tz = LithosphericCooling(Age,z/1000.,'GDH1')
    geotherms = Tz.T
    CurieDepthByAge = np.zeros((Age.shape[0]))
    for i in np.arange(0,Age.shape[0]):
        CDInterpolator = interp1d(geotherms[i,:],z/1000.,kind='linear',\
                                          bounds_error=False,fill_value=0)
        CurieDepthByAge[i] = CDInterpolator(CurieTemp)


    TTCDInterpolator = interp1d(CurieDepthByAge,Age,kind='linear',\
                                          bounds_error=False,fill_value=0)
    TimeToCurieByDepth = TTCDInterpolator(z/1000.)

    # For Depth values where the Curie Temperature is not reached for any 
    # time in the time array, set to 400
    TimeToCurieByDepth[1+np.where(TimeToCurieByDepth==np.max(TimeToCurieByDepth))[0][0]:] = 400.
    
    return CurieDepthByAge*1000.,TimeToCurieByDepth


def age2polarity(Age, timescalefile=DEFAULT_TIMESCALE_FILE):
    # The input is assumed to be a numpy array. 
    # It can contains NaNs, which will be returned as NaNs while the polarities will be plus or minus one
    
    TimeScaleArray = np.genfromtxt(timescalefile,usecols=(0,)) 

    PolarityGrid = np.zeros(Age.shape)*np.nan

    for t in range(0,len(TimeScaleArray)-1,2):
        ind = np.where(np.logical_and(Age>TimeScaleArray[t],Age<TimeScaleArray[t+1]))
        PolarityGrid[ind]=1
    for t in range(1,len(TimeScaleArray)-1,2):
        ind = np.where(np.logical_and(Age>TimeScaleArray[t],Age<TimeScaleArray[t+1]))
        PolarityGrid[ind]=-1
        
    return PolarityGrid
        

def agearray2magnetisation(AgeGrid,Age,RM):
    # Use the remanent magnetisation as a function of age to get the magnetisation
    # spatially by mapping to the age grid.
    MInterpolator = interp1d(Age,RM,kind='linear',\
                             bounds_error=False,fill_value=0)
    M = MInterpolator(AgeGrid)
    
    return M


def agearray2magneticlayerthickness(Age,CurieDepthByAge,AgeGrid):
    # Also get a map of the oceanic lithosphere magnetic thickness (ie the thickness
    # that is cooler than the Curie Depth). 
    OCInterpolator = interp1d(Age,CurieDepthByAge,kind='linear',\
                              bounds_error=False,fill_value=0)
    OCThickness = OCInterpolator(AgeGrid)
    
    return OCThickness


def paleoIncDec2field(PLat,Dec):
    
    Inc = np.arctan(2* np.tan(PLat*np.pi/180) ) *180/np.pi

    # Note the annoying 'gotcha' here - the equations are taken from Blakely's
    # textbook, BUT - in the book, the x direction is defined as north, and y is to
    # the east. This code uses x is east convention throughout.
    Bx = np.cos(Inc*np.pi/180)*np.cos(Dec*np.pi/180)
    By = np.cos(Inc*np.pi/180)*np.sin(Dec*np.pi/180)
    Bz = np.sin(Inc*np.pi/180)
    Btheta = -Bx
    Bphi = By
    Br = -Bz

    return Br,Btheta,Bphi


def vim2magnetisation(AgeGrid,PLat,Dec,Age,RM):

    Br,Btheta,Bphi = paleoIncDec2field(PLat,Dec)

    # NB in Masterton et al it has sin of colatitude, but Dyment & Arkani-Hamed,
    # and Hemant & Maus, have it different
    C = np.sqrt(1+(3*np.sin((PLat)*np.pi/180)**2))

    M = agearray2magnetisation(AgeGrid,Age,RM)

    # Multiply the Magnetisation by layer thickness, factor for latitude dependence
    # of amplitude, and the three components of the magnetising field
    Ocean_RVIM_Mr = M*C*Br
    Ocean_RVIM_Mtheta = M*C*Btheta
    Ocean_RVIM_Mphi = M*C*Bphi

    return Ocean_RVIM_Mr, Ocean_RVIM_Mtheta, Ocean_RVIM_Mphi
    

def vis2magnetisation(OceanVIS,hires=False):
        
    # Next we calculate the VIM components using the contemporary magnetising field,
    # precalculated at the appropriate resolution
    if hires:
        _,_,Bx = load_netcdf('../input_grids/IGRF_Bx_6m.nc')
        _,_,By = load_netcdf('../input_grids/IGRF_By_6m.nc')
        _,_,Bz = load_netcdf('../input_grids/IGRF_Bz_6m.nc')        
    else:
        _,_,Bx = load_netcdf('../input_grids/IGRF_Bx.nc')
        _,_,By = load_netcdf('../input_grids/IGRF_By.nc')
        _,_,Bz = load_netcdf('../input_grids/IGRF_Bz.nc')

    Btheta = -1*Bx
    Bphi = By
    Br = -1*Bz
    Ocean_IVIM_Mr = OceanVIS*Br
    Ocean_IVIM_Mtheta = OceanVIS*Btheta
    Ocean_IVIM_Mphi = OceanVIS*Bphi
    
    return Ocean_IVIM_Mr, Ocean_IVIM_Mtheta, Ocean_IVIM_Mphi 


def resample_global_grid(Xg,Yg,Grid,Sampling=0.25):
    # These are the coordinates of the grid we want to end up with, for use later on
    # And note the funny order, to aid the writing to the 'total.in' file
    gridX = np.arange(0,360,Sampling)
    gridY = np.arange(Sampling/2.,180,Sampling)
    gridXX,gridYY = np.meshgrid(gridX,gridY)

    # using pyresample: define the coordinates we want to sample onto
    area_def = geometry.GridDefinition(lons=gridXX-180, \
                                       lats=gridYY-90)

    # define the coordinates we want to sample from
    Xg,Yg = np.meshgrid(Xg,Yg)
    grid_def = geometry.GridDefinition(lons=Xg, \
                                       lats=Yg)


    msg_con_nn = image.ImageContainerNearest(Grid, grid_def, radius_of_influence=50000)
    area_con_nn = msg_con_nn.resample(area_def)
    ResampledGrid = area_con_nn.image_data
    
    dateline_index = np.int(gridX.shape[0]/2)
    ResampledGrid = np.hstack((ResampledGrid[:,dateline_index:],ResampledGrid[:,1:dateline_index+1]))
    
    return gridX,gridY,ResampledGrid


def MergeContinentsWithOceans(OceanGrid,ContinentGrid):
    
    tmp = OceanGrid
    #tmp_data = np.ma.getdata(tmp)
    #tmp_mask = np.ma.getmask(tmp)
    tmp_mask = np.isnan(tmp)
    
    tmp3 = np.zeros(tmp.shape)
    tmp5 = np.zeros(tmp.shape)
    
    #tmp4 = np.flipud(ContinentGrid)
    tmp3[tmp_mask] = ContinentGrid[tmp_mask]
    tmp5[np.logical_not(tmp_mask)] = tmp[np.logical_not(tmp_mask)]
    tmp6 = tmp3+tmp5
    
    return tmp6


#-----------------------------------------------------------------------
# Functions for File IO
#-----------------------------------------------------------------------
def load_netcdf(grdfile,z_field_name='z'):
    
    ds_disk = xr.open_dataset(grdfile)

    data_array = ds_disk[z_field_name]
    coord_keys = [key for key in data_array.coords.keys()]  # updated for python3 compatibility
    

    if 'lon' in coord_keys[0].lower():
        latitude_key=1; longitude_key=0
    elif 'x' in coord_keys[0].lower():
        latitude_key=1; longitude_key=0
    else:
        latitude_key=0; longitude_key=1

    try:
        gridX = data_array.coords[coord_keys[longitude_key]].data
        gridY = data_array.coords[coord_keys[latitude_key]].data
        gridZ = data_array.data
    except:
        # attempt to handle old-school GMT netcdfs (e.g. produced by grdconvert)
        gridX = np.linspace(ds_disk.data_vars['x_range'].data[0],
                            ds_disk.data_vars['x_range'].data[1],
                            ds_disk.data_vars['dimension'].data[0])
        gridY = np.linspace(ds_disk.data_vars['y_range'].data[0],
                            ds_disk.data_vars['y_range'].data[1],
                            ds_disk.data_vars['dimension'].data[1])
        gridZ = np.flipud(ds_disk.data_vars[z_field_name].data.reshape(ds_disk.data_vars['dimension'].data[1],
                                                                       ds_disk.data_vars['dimension'].data[0]))

    ds_disk.close()
    
    if gridZ.shape[0]==gridX.shape[0]:
        gridZ = gridZ.T
    
    return gridX,gridY,gridZ


def WriteCDFGrid(filename,x,y,z):
    ds = xr.DataArray(z, coords=[('y',y), ('x',x)], name='z')
    ds.to_netcdf(filename, format='NETCDF4')


#-----------------------------------------------------------------------
# Function for points in polygons tests
#-----------------------------------------------------------------------
def assign_polygons_to_spatial_tree(static_polygon_features, uniform_points, spatial_tree_of_uniform_points):
    # Extract the polygons and plate IDs from the reconstructed static polygons.
    static_polygons = []
    static_polygon_plate_ids = []
    for static_polygon_feature in static_polygon_features:
        plate_id = static_polygon_feature.get_reconstruction_plate_id()
        polygon = static_polygon_feature.get_geometry()

        static_polygon_plate_ids.append(plate_id)
        static_polygons.append(polygon)

    #print 'Find static polygons...'

    # Find the reconstructed static polygon (plate IDs) containing the uniform (reconstructed) points.
    #
    # The order (and length) of 'recon_point_plate_ids' matches the order (and length) of 'unhttps://feedly.com/i/latestiform_recon_points'.
    # Points outside all static polygons return a value of None.
    point_plate_ids = points_in_polygons.find_polygons_using_points_spatial_tree(
        uniform_points, spatial_tree_of_uniform_points, static_polygons, static_polygon_plate_ids)

    return point_plate_ids, static_polygons, static_polygon_plate_ids
 
    
def group_points_by_polygons(point_plate_ids,uniform_points):
    #print 'Group by polygons...'

    # Group recon points with plate IDs so we can later create one multipoint per plate.
    points_grouped_by_plate_id = {}
    for point_index, point_plate_id in enumerate(point_plate_ids):
        # Reject any points outside all reconstructed static polygons.
        if point_plate_id is None:
            continue

        # Add empty list to dict if first time encountering plate ID.
        if point_plate_id not in points_grouped_by_plate_id:
            points_grouped_by_plate_id[point_plate_id] = []

        # Add to list of points associated with plate ID.
        point = uniform_points[point_index]
        points_grouped_by_plate_id[point_plate_id].append(point)

    return points_grouped_by_plate_id
        
    
def group_age_coded_points_by_polygons(point_plate_ids,uniform_points,point_ages):
    #print 'Group by polygons...'

    # Group recon points with plate IDs so we can later create one multipoint per plate.
    points_grouped_by_plate_id = {}
    for point_index, point_plate_id in enumerate(point_plate_ids):
        # Reject any points outside all reconstructed static polygons.
        if point_plate_id is None:
            continue

        # Add empty list to dict if first time encountering plate ID.
        if point_plate_id not in points_grouped_by_plate_id:
            points_grouped_by_plate_id[point_plate_id] = []

        # Add to list of points associated with plate ID.
        point = uniform_points[point_index]
        point_age = point_ages[point_index]
        points_grouped_by_plate_id[point_plate_id].append((point,point_age))

    return points_grouped_by_plate_id


def lat_long_2_spatial_tree(lons,lats):
    
    uniform_points = [pygplates.PointOnSphere(lat, lon) for lat, lon in zip(lats,lons)]
    
    spatial_tree_of_uniform_points = points_spatial_tree.PointsSpatialTree(uniform_points)
                                                                            
    return uniform_points,spatial_tree_of_uniform_points


def multipart_2_singlepart(multipart_features):
    singlepart_features = []
    for multipart_feature in multipart_features:
        for geometry in multipart_feature.get_geometries():
            feature = multipart_feature
            feature.set_geometry(geometry)
            singlepart_features.append(feature)
            
    return singlepart_features


def fill_grid_points_within_polygons(grid,uniform_points,spatial_tree_of_uniform_points,masking_polygon_filename):

    masking_polygon_features = pygplates.FeatureCollection(masking_polygon_filename)

    gridZ_filled = inpaint.fill_ndimage(grid).flatten()

    fill_points = np.zeros(len(uniform_points))*np.nan

    for masking_polygon_feature in masking_polygon_features:

        (point_plate_ids,
         static_polygons,
         static_polygon_plate_ids) = assign_polygons_to_spatial_tree([masking_polygon_feature], 
                                                                     uniform_points, 
                                                                     spatial_tree_of_uniform_points)

        ind = np.where(point_plate_ids)
        fill_points[ind] = gridZ_filled[ind]


    fill_raster = np.reshape(fill_points,grid.shape)

    grid[np.isnan(grid)] = fill_raster[np.isnan(grid)]

    return grid


def EulerRotation(Xo,Yo,rAngle,DataX,DataY):

    #%%%%%%%%%%%%%%%%%%
    # Build Rotation Matrix
    Xo = Xo*np.pi/180;
    Yo = Yo*np.pi/180;
    rAngle = rAngle*np.pi/180;

    Ex = np.cos(Yo) * np.cos(Xo);
    Ey = np.cos(Yo) * np.sin(Xo);
    Ez = np.sin(Yo);

    R11 = Ex*Ex * (1-np.cos(rAngle)) + np.cos(rAngle);
    R12 = Ex*Ey * (1-np.cos(rAngle)) - Ez*np.sin(rAngle);
    R13 = Ex*Ez * (1-np.cos(rAngle)) + Ey*np.sin(rAngle);

    R21 = Ey*Ex * (1-np.cos(rAngle)) + Ez*np.sin(rAngle);
    R22 = Ey*Ey * (1-np.cos(rAngle)) + np.cos(rAngle);
    R23 = Ey*Ez * (1-np.cos(rAngle)) - Ex*np.sin(rAngle);

    R31 = Ez*Ex * (1-np.cos(rAngle)) - Ey*np.sin(rAngle);
    R32 = Ez*Ey * (1-np.cos(rAngle)) + Ex*np.sin(rAngle);
    R33 = Ez*Ez * (1-np.cos(rAngle)) + np.cos(rAngle);

    #%%%%%%%%%%%%%%%%%%
    #% Rotate Data

    #% convert to radians
    DataX = DataX*np.pi/180;
    DataY = DataY*np.pi/180;

    #% convert to cartesian coordinates
    Ax = np.cos(DataY)*np.cos(DataX);
    Ay = np.cos(DataY)*np.sin(DataX);
    Az = np.sin(DataY);

    #% Apply rotation
    Axr = R11*Ax + R12*Ay + R13*Az;
    Ayr = R21*Ax + R22*Ay + R23*Az;
    Azr = R31*Ax + R32*Ay + R33*Az;

    #% Convert back to Long/Lat
    #% Assumes that we want -180 to +180 Long/Lats
    #DataXR = zeros(size(DataX));
    if (Axr==0 and Ayr<0):
        DataXR = -90;
    elif (Axr==0 and Ayr>=0):
        DataXR = 90;
    elif (Axr<0 and Ayr<0): 
        DataXR = -180+(np.arctan(Ayr/Axr)*180/np.pi);
    elif (Axr<0 and Ayr>=0): 
        DataXR = 180+(np.arctan(Ayr/Axr)*180/np.pi);
    else:   #if (Axr>0); 
        DataXR = np.arctan(Ayr/Axr)*180/np.pi;

    DataYR = np.arcsin(Azr)*180/np.pi;

    return DataXR,DataYR


def generate_healpix_points(N):
    othetas,ophis = hp.pix2ang(N,np.arange(12*N**2))
    othetas = np.pi/2-othetas
    ophis[ophis>np.pi] -= np.pi*2

    # ophis -> longitude, othetas -> latitude
    longitude = np.degrees(ophis)
    latitude = np.degrees(othetas)
    
    return longitude,latitude
    
