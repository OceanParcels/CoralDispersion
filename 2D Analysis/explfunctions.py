# -*- coding: utf-8 -*-
"""
Exploratory functions

Created on Thu Nov 21 12:44:03 2019

@author: reint fischer
"""
import math
from parcels import JITParticle, Variable
from operator import attrgetter
import numpy as np
from netCDF4 import Dataset
import xarray as xr

def AdvectionRK4_3D_v(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.
    
    Adapted from Parcels code, this Kernel saves the velocity components used to calculate the next location.

    Function needs to be converted to Kernel object before execution"""
    (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
    lon1 = particle.lon + u1*.5*particle.dt
    lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]
    lon2 = particle.lon + u2*.5*particle.dt
    lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]
    lon3 = particle.lon + u3*particle.dt
    lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]
    U = (u1 + 2*u2 + 2*u3 + u4) / 6.
    V = (v1 + 2*v2 + 2*v3 + v4) / 6.
    W = (w1 + 2*w2 + 2*w3 + w4) / 6.
    particle.u = U
    particle.v = V
    particle.w = W
    particle.lon += U * particle.dt
    particle.lat += V * particle.dt
    particle.depth += W * particle.dt
    
def deleteparticle(particle,fieldset,time):
    """ This function deletes particles as they exit the domain and prints a message about their attributes at that moment
    """
    
    print('Particle '+str(particle.id)+' has died at t = '+str(time)+' at lon, lat, depth = '+str(particle.lon)+', '+str(particle.lat)+', '+str(particle.depth))
    particle.delete()


def removeNaNs(particle,fieldset,time):
    """ This function removes the masked particles
    """
    if particle.lon == 0 and particle.depth==0:
        particle.delete()

class DistParticle(JITParticle):  # Define a new particle class that contains three extra variables
    finaldistance = Variable('finaldistance', initial=0., dtype=np.float32)  # the distance travelled
    prevlon = Variable('prevlon', dtype=np.float32, to_write=False,
                        initial=attrgetter('lon'))  # the previous longitude
    prevlat = Variable('prevlat', dtype=np.float32, to_write=False,
                        initial=attrgetter('lat'))  # the previous latitude.
    prevdepth = Variable('prevdepth', dtype=np.float32, to_write=False,
                        initial=attrgetter('depth'))  # the previous latitude.
#     d2c = Variable('d2c', dtype=np.float32, initial=fieldset.dist)
#     u = Variable('u', dtype=np.float32, initial=0.)  # velocity in x-direction
#     v = Variable('v', dtype=np.float32, initial=0)  # velocity in x-direction
#     w = Variable('w', dtype=np.float32, initial=0)  # velocity in x-direction

def FinalDistance(particle,fieldset,time):
    lat_dist = particle.lat-particle.prevlat
    lon_dist = particle.lon-particle.prevlon
    depth_dist = particle.depth-particle.prevdepth
    
    particle.finaldistance = math.sqrt(math.pow(lon_dist, 2) + math.pow(lat_dist, 2)  + math.pow(depth_dist, 2))
    particle.prevlat = particle.lat
    particle.prevlon = particle.lon
    particle.prevdepth = particle.depth

def removeBeached(particle,fieldset,time):
    if particle.lon == particle.prevlon and particle.lat == particle.prevlat:
        print('Particle '+str(particle.id)+' has beached at t = '+str(time)+' at lon, lat, depth = '+str(particle.lon)+', '+str(particle.lat)+', '+str(particle.depth))
        particle.delete()

def coraldistancemap(coralmask,edgemask,x_mesh,y_mesh):
    """ Function generating a map with the minimum distances to a coral object
    """
#     flatx_mesh = x_mesh.flatten()
#     flaty_mesh = y_mesh.flatten()
    
    coralcoords = np.asarray((x_mesh[~coralmask],y_mesh[~coralmask]))
    edgecoords = np.asarray((x_mesh[edgemask],y_mesh[edgemask]))
    D2coralmatrix = np.zeros((len(edgecoords[0]),len(coralcoords[0])))
    coralmap = np.zeros(x_mesh.shape)
    
    for i in range(len(edgecoords[0])):
        D2coralmatrix[i,:] = np.sqrt(np.power(coralcoords[0,:]-edgecoords[0,i],2)+np.power(coralcoords[1,:]-edgecoords[1,i],2))
    
    coralmap[~coralmask] = np.min(D2coralmatrix,axis=0)
    return D2coralmatrix,coralmap

def followPath(i,j,coastmask,object1):
    x = 0
    y = 0
    end = 0
    
    while end<1:
        if -1<i+x<len(coastmask) and -1<j+y+1<len(coastmask[i]) and coastmask[i+x,j+y+1] == True:
            object1[i+x,j+y+1] = True
            coastmask[i+x,j+y+1] = False
            y += 1
        elif -1<i+x-1<len(coastmask) and -1<j+y+1<len(coastmask[i]) and coastmask[i+x-1,j+y+1] == True:
            object1[i+x-1,j+y+1] = True
            coastmask[i+x-1,j+y+1] = False
            x += -1
            y += 1
        elif -1<i+x+1<len(coastmask) and -1<j+y+1<len(coastmask[i]) and coastmask[i+x+1,j+y+1] == True:
            object1[i+x+1,j+y+1] = True
            coastmask[i+x+1,j+y+1] = False
            x += 1
            y += 1
        elif -1<i+x+1<len(coastmask) and -1<j+y<len(coastmask[i]) and coastmask[i+x+1,j+y] == True:
            object1[i+x+1,j+y] = True
            coastmask[i+x+1,j+y] = False
            x += 1
        elif -1<i+x-1<len(coastmask) and -1<j+y<len(coastmask[i]) and coastmask[i+x-1,j+y] == True:
            object1[i+x-1,j+y] = True
            coastmask[i+x-1,j+y] = False
            x += -1
        elif -1<i+x-1<len(coastmask) and -1<j+y-1<len(coastmask[i]) and coastmask[i+x-1,j+y-1] == True:
            object1[i+x-1,j+y-1] = True
            coastmask[i+x-1,j+y-1] = False
            x += -1
            y += -1
        elif -1<i+x<len(coastmask) and -1<j+y-1<len(coastmask[i]) and coastmask[i+x,j+y-1] == True:
            object1[i+x,j+y-1] = True
            coastmask[i+x,j+y-1] = False
            y += -1
        elif -1<i+x+1<len(coastmask) and -1<j+y-1<len(coastmask[i]) and coastmask[i+x+1,j+y-1] == True:
            object1[i+x+1,j+y-1] = True
            coastmask[i+x+1,j+y-1] = False
            x += 1
            y += -1
        else:
            end += 1
    return coastmask, object1, x, y

# def checkSameObject(object1,objects,coastmask)