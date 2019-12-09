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

def deleteparticle(particle,fieldset,time):
    print('Particle '+str(particle.id)+' has died at t = '+str(time)+' at lon, lat, depth = '+str(particle.lon)+', '+str(particle.lat)+', '+str(particle.depth))
    particle.delete()


def removeNaNs(particle,fieldset,time):
    if particle.lon == 0 and particle.depth==0:
        particle.delete()

class DistParticle(JITParticle):  # Define a new particle class that contains three extra variables
    finaldistance = Variable('finaldistance', initial=0., dtype=np.float32)  # the distance travelled
    prevlon = Variable('prevlon', dtype=np.float32, to_write=False,
                        initial=attrgetter('lon'))  # the previous longitude
    prevlat = Variable('prevlat', dtype=np.float32, to_write=False,
                        initial=attrgetter('lat'))  # the previous latitude.

def FinalDistance(particle,fieldset,time):
    lat_dist = particle.lat-particle.prevlat
    lon_dist = particle.lon-particle.prevlon
    particle.finaldistance = math.sqrt(math.pow(lon_dist, 2) + math.pow(lat_dist, 2))
    particle.prevlat = particle.lat
    particle.prevlon = particle.lon

def removeBeached(particle,fieldset,time):
    if particle.lon == particle.prevlon and particle.lat == particle.prevlat:
        print('Particle '+str(particle.id)+' has beached at t = '+str(time)+' at lon, lat, depth = '+str(particle.lon)+', '+str(particle.lat)+', '+str(particle.depth))
        particle.delete()

def selectBeached(filename):
    nc = Dataset(filename+fb+str(runtime.seconds)+".nc")
    x = nc.variables["lon"][:].squeeze()
    y = nc.variables["lat"][:].squeeze()
    z = nc.variables["z"][:].squeeze()
    t = nc.variables["time"][:].squeeze()
    b = nc.variables["finaldistance"][:].squeeze()
    nc.close()
    
    mask = np.nonzero(b[:,-1]==0)[0]
    xn = x[mask]
    yn = y[mask]
    zn = z[mask]
    tn = t[mask]
    bn = b[mask]
    return xn,yn,zn,tn,bn