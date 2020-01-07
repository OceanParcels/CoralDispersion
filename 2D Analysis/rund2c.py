# -*- coding: utf-8 -*-
"""
Parcels run sampling distance from coral

Created on Thu Dec 12 14:09:03 2019

@author: reint fischer
"""

import sys
sys.path.insert(0, "\\Users\\Gebruiker\\Documents\\GitHub\\parcels\\")  # Set path to find the newest parcels code

from parcels import FieldSet, ParticleSet, Variable, JITParticle, ScipyParticle, AdvectionRK4_3D, ErrorCode, plotTrajectoriesFile
import numpy as np
from operator import attrgetter
from datetime import timedelta
from netCDF4 import Dataset,num2date,date2num
from explfunctions import deleteparticle, removeNaNs, DistParticle, FinalDistance

filename = 'output-corals-regridded'
fb = 'forward' #variable to determine whether the flowfields are analysed 'forward' or 'backward' in time
corald = Dataset(filename+'.nc','r+') # read netcdf file with input

# Extract all variables into np arrays --> in the future xarray will be used
T = corald.variables['T'][:]
X = corald.variables['X'][:]
Y = corald.variables['Y'][:]
U = corald.variables['U'][:]
V = corald.variables['V'][:]

corald.close()

U = np.asarray(U)
U = np.expand_dims(U,2)            # add a third dimension

V = np.asarray(V)
V = np.expand_dims(V,2)            # add a third dimension

t = num2date(T,units='seconds since 2000-01-01 00:00:00.0') # make t a datetime object
t = date2num(t,units='seconds since 2000-01-01 00:00:00.0')

times = t
xs = X
ys = np.asarray([-1,0,1])          # third dimension with length 3. 2D flow field will be inserted on the middle value to ensure the AdvectionRK4_3D works correctly
depths = -Y                        # Y was height, but parcels expects depth

u = np.zeros(U.shape)
u = np.concatenate((u,u,u),axis=2) # add the third dimension
u[:,:,1,:] = U[:,:,0,:]            # add the data to the middle value of the third dimension
v = np.zeros(u.shape)
w = np.zeros(U.shape)
w = np.concatenate((w,w,w),axis=2) # add the third dimension
w[:,:,1,:] = -V[:,:,0,:]           # because depth = -Y, w = -V
dist = np.zeros(u.shape)
coralmap = np.load('coralmap.npy')
dist[:,:,1,:] = np.asarray([coralmap]*len(u))
closest = np.zeros(u.shape)
closestobject = np.load('closestobject.npy')
# index = np.ma.masked_array(closestobject,umask.mask)
closest[:,:,1,:] = np.asarray([closestobject]*len(u))

data = {'U': u,
        'V': v,
        'W': w,
        'D': dist,
        'C': closest}
dimensions = {'lon':xs,
              'lat':ys,
              'depth':depths,
              'time':times}
fieldset = FieldSet.from_data(data=data, dimensions= dimensions, mesh='flat')
fieldset.C.interp_method = 'nearest'

class DistParticle(JITParticle):  # Define a new particle class that contains three extra variables
    finaldistance = Variable('finaldistance', initial=0., dtype=np.float32)  # the distance travelled
    prevlon = Variable('prevlon', dtype=np.float32, to_write=False,
                        initial=attrgetter('lon'))  # the previous longitude
    prevlat = Variable('prevlat', dtype=np.float32, to_write=False,
                        initial=attrgetter('lat'))  # the previous latitude.
    prevdepth = Variable('prevdepth', dtype=np.float32, to_write=False,
                        initial=attrgetter('depth'))  # the previous latitude.
    d2c = Variable('d2c', dtype=np.float32, initial=0.)
    closestobject = Variable('closestobject', dtype=np.float32, initial=0.)
    
    
def SampleD(particle, fieldset, time):  # Custom function that samples fieldset.P at particle location
    particle.d2c = fieldset.D[time, particle.depth, particle.lat, particle.lon]
    particle.closestobject = fieldset.C[time, particle.depth, particle.lat, particle.lon]
    
lons, ds = np.meshgrid(xs,depths)                        # meshgrid at all gridpoints in the flow data
um = np.ma.masked_invalid(u[0,:,1,:])                    # retrieve mask from flowfield to take out points over coral objects

lons = np.ma.masked_array(lons,mask=um.mask)             # mask points in meshgrid
lons = lons.flatten()
ds = np.ma.masked_array(ds,mask=um.mask)                 # mask points in meshgrid
ds = ds.flatten()

outputdt = timedelta(seconds=0.1)                        # timesteps to create output at
dt=timedelta(seconds=0.01)                               # timesteps to calculate particle trajectories
runtime=timedelta(seconds=44)                            # total time to execute the particleset
lats = np.asarray([0]*len(lons))                         # all particles must start and stay on the middle value of the extra dimension
inittime = np.asarray([0]*len(lons))                     # default time to start the particles is zero
if fb == 'backward':                                     # change timestep and start time when in 'backward' mode
    dt = dt*-1
    inittime = np.asarray([runtime.seconds]*len(lons))
    
pset = ParticleSet(fieldset=fieldset, pclass=DistParticle, lon=lons, lat=lats, depth=ds,time=inittime)
n_part = pset.size

k_removeNaNs = pset.Kernel(removeNaNs)
k_sample = pset.Kernel(SampleD)    # Casting the SampleP function to a kernel.

pset.execute(k_removeNaNs+k_sample, runtime = timedelta(seconds=0))

k_dist = pset.Kernel(FinalDistance)  # Casting the FinalDistance function to a kernel.

pset.execute(AdvectionRK4_3D+k_dist+k_sample,
             runtime=runtime,
             dt=dt,
             recovery = {ErrorCode.ErrorOutOfBounds:deleteparticle},
             output_file=pset.ParticleFile(name='d2c'+filename+fb+str(runtime.seconds), outputdt=outputdt)
            )