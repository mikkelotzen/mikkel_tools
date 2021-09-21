import numpy as np

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