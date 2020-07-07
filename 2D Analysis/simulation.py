# -*- coding: utf-8 -*-
"""
Parcels run sampling distance from coral

Created on Thu Dec 12 14:09:03 2019

@author: reint fischer
"""

import sys
sys.path.insert(0, "\\Users\\Gebruiker\\Documents\\GitHub\\parcels\\")  # Set path to find the newest parcels code
import time as ostime
import numpy as np

from parcels import FieldSet, ParticleSet, AdvectionRK4_3D, ErrorCode
from datetime import timedelta
from netCDF4 import Dataset,num2date,date2num

from functions import deleteparticle, removeNaNs, DistParticle, FinalDistance, Samples, boundary_advectionRK4_3D

def run(flow,dt,bconstant,foldername ='21objects'):
    DistParticle.setLastID(0)
    filename = flow
    fb = 'forward' # variable to determine whether the flowfields are analysed 'forward' or 'backward' in time
    bconstant = bconstant

    corald = Dataset(foldername+'/'+filename+'.nc','r+') # read netcdf file with input

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
    t = num2date(T[:61],units='seconds since 2000-01-01 00:00:00.0') # make t a datetime object
    t = date2num(t,units='seconds since 2000-01-01 00:00:00.0')

    times = t
    xs = X
    ys = np.asarray([-1,0,1])          # third dimension with length 3. 2D flow field will be inserted on the middle value to ensure the AdvectionRK4_3D works correctly
    depths = -Y                        # Y was height, but parcels expects depth

    u = np.zeros((61,U.shape[1],U.shape[2],U.shape[3]))
    u = np.concatenate((u,u,u),axis=2) # add the third dimension
    u[:,:,1,:] = U[:61,:,0,:]            # add the data to the middle value of the third dimension
    v = np.zeros(u.shape)
    w = np.zeros(u.shape)
    w[:,:,1,:] = -V[:61,:,0,:]           # because depth = -Y, w = -V

    dist = np.zeros(u.shape)
    distancemap = np.load(foldername + '/preprocessed/' + 'distancemap.npy')
    dist[:,:,1,:] = np.asarray([distancemap] * len(u))

    closest = np.zeros(u.shape)
    closestobject = np.load(foldername+'/preprocessed/'+'closestobject.npy')
    closest[:,:,1,:] = np.asarray([closestobject]*len(u))

    border = np.zeros(u.shape)
    bordermap = np.load(foldername+'/preprocessed/'+'bordermap.npy')
    border[:,:,1,:] = np.asarray([bordermap]*len(u))

    data = {'U': u,
            'V': v,
            'W': w,
            'B': border,
            'C': closest,
            'D': dist}
    dimensions = {'lon':xs,
                  'lat':ys,
                  'depth':depths,
                  'time':times}

    fieldset = FieldSet.from_data(data=data, dimensions= dimensions, mesh='flat')
    fieldset.B.interp_method = 'nearest'
    fieldset.C.interp_method = 'nearest'
    fieldset.add_constant('dx', X[1]-X[0])
    fieldset.add_constant('beaching', bconstant)
    fieldset.add_constant('x0',xs[0])
    fieldset.add_constant('y0',ys[0])
    fieldset.add_constant('z0',depths[0])

    lons, ds = np.meshgrid(xs,depths[:])                        # meshgrid at all gridpoints in the flow data
    um = np.ma.masked_invalid(u[0,:,1,:])                    # retrieve mask from flowfield to take out points over coral objects

    lons = np.ma.masked_array(lons,mask=um.mask[:,:])             # mask points in meshgrid
    lons = np.ma.filled(lons,-999)
    lons = lons.flatten()
    ds = np.ma.masked_array(ds,mask=um.mask[:,:])                 # mask points in meshgrid
    ds = np.ma.filled(ds,-999)
    ds = ds.flatten()

    outputdt = timedelta(seconds=0.1)                        # timesteps to create output at
    dt=timedelta(seconds=dt)                              # timesteps to calculate particle trajectories
    runtime=timedelta(seconds=60)                            # total time to execute the particleset
    lats = np.asarray([0]*len(lons))                         # all particles must start and stay on the middle value of the extra dimension
    inittime = np.asarray([0]*len(lons))                     # default time to start the particles is zero
    if fb == 'backward':                                     # change timestep and start time when in 'backward' mode
        dt = dt*-1
        inittime = np.asarray([runtime.seconds]*len(lons))

    pset = ParticleSet(fieldset=fieldset, pclass=DistParticle, lon=lons, lat=lats, depth=ds,time=inittime)
    n1_part = pset.size

    k_removeNaNs = pset.Kernel(removeNaNs)
    k_sample = pset.Kernel(Samples)    # Casting the SampleP function to a kernel.

    pset.execute(k_removeNaNs+k_sample, runtime = timedelta(seconds=0))
    n2_part = pset.size

    k_dist = pset.Kernel(FinalDistance)  # Casting the FinalDistance function to a kernel.
    k_bound = pset.Kernel(boundary_advectionRK4_3D)  # Casting the Boundary_Advection function to a kernel.

    output_file=pset.ParticleFile(name=foldername+'/pfiles/B'+str(fieldset.beaching)+'-'+flow+'-'+str(abs(dt.total_seconds()))[2:]+'-'+fb, outputdt=outputdt)

    stime = ostime.time()
    pset.execute(k_bound + k_dist + k_sample,
                 runtime=runtime,
                 dt=dt,
                 recovery = {ErrorCode.ErrorOutOfBounds:deleteparticle},
                 output_file=output_file)
    etime = ostime.time()

    output_file.add_metadata('outputdt',str(outputdt.total_seconds())+' in seconds')
    output_file.add_metadata('runtime',str(runtime.total_seconds())+' in seconds')
    output_file.add_metadata('dt',str(dt.total_seconds())+' in seconds')
    output_file.add_metadata('dx', float(np.abs(X[1]-X[0])))
    output_file.add_metadata('executiontime',str(etime-stime)+' in seconds')
    output_file.add_metadata('beaching_strategy',fieldset.beaching)

    output_file.close()

    n3_part = pset.size
    print('Amount of particles at initialisation, 0th timestep and after execution respectively:'+str(n1_part)+', '+str(n2_part)+', '+str(n3_part))

if __name__ == "__main__":
    run('waveparabolic', 0.001, 2)





