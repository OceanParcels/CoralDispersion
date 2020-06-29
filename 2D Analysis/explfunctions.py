# -*- coding: utf-8 -*-
"""
Functions for the master thesis on Lagrangian connectivity over a complex coral topography

Created on Thu Nov 21 12:44:03 2019

@author: reint fischer
"""
import math
from parcels import JITParticle, Variable, ScipyParticle
from operator import attrgetter
import numpy as np

def boundary_advectionRK4_3D(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.
    
    Adapted from Parcels code, this Kernel interpolates the velocities differently close to the coral objects.
    
    Needs to be used in combination with the DistParticle class

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

    # Determine final velocities to be checked against the directions of the boundaries
    u_final = (u1 + 2 * u2 + 2 * u3 + u4) / 6.
    v_final = (v1 + 2 * v2 + 2 * v3 + v4) / 6.
    w_final = (w1 + 2 * w2 + 2 * w3 + w4) / 6.

    B_binary = particle.border
    n_border = 0

    f_u = 1
    lon_cell = (particle.lon - fieldset.x0 + 0.5*fieldset.dx)/fieldset.dx
    lon_frac = lon_cell - math.floor(lon_cell)
    f_v = 1
    lat_cell = (particle.lat - fieldset.y0 + 0.5*fieldset.dx) / fieldset.dx
    lat_frac = lat_cell - math.floor(lat_cell)
    f_w = 1
    depth_cell = (particle.depth - fieldset.z0 + 0.5*fieldset.dx) / fieldset.dx
    depth_frac = depth_cell - math.floor(depth_cell)

    frac_d2c = particle.d2c / fieldset.dx
    f_linear1 = 2 * frac_d2c - 1
    f_sq1 = 4*math.pow(frac_d2c -0.5,2)
    f_exp1 = math.pow(10,-math.log(0.5,base=10) * frac_d2c - math.log(0.5,base=10))-3
    f_cub1 = 8*math.pow(frac_d2c -0.5,3)

    if fieldset.beaching == 1: # linear - linear
        if B_binary >= 8: # border right
            B_binary += -8
            if lon_frac > 0.5:
                f_u = 2 - 2 * lon_frac
                f_w = 2 - 2 * lon_frac
        if B_binary >= 4: # border left
            B_binary += -4
            if lon_frac < 0.5:
                f_u = 2 * lon_frac
                f_w = 2 * lon_frac
        if B_binary >= 2: # border below
            B_binary += -2
            if depth_frac > 0.5:
                f_u = 2- 2 * depth_frac
                f_w = 2- 2 * depth_frac
        if B_binary == 1:  # border above
            if depth_frac < 0.5:
                f_u = 2 * depth_frac
                f_w = 2 * depth_frac

    if fieldset.beaching == 2: # linear - sqrt
        if B_binary >= 8:  # border right
            B_binary += -8
            if lon_frac > 0.5:
                f_u = 4 * math.pow(1-lon_frac, 2)  # no cross-boundary movement
                f_w = 2 - 2 * lon_frac
                n_border += 1
        if B_binary >= 4:  # border left
            B_binary += -4
            if lon_frac < 0.5 and n_border == 0:
                f_u = 4 * math.pow(lon_frac, 2)
                f_w = 2 * lon_frac
                n_border += 1
        if B_binary >= 2:  # border below
            B_binary += -2
            if depth_frac > 0.5 and n_border == 0:
                f_u = 2 - 2 * depth_frac
                f_w = 4 * math.pow(1 - depth_frac, 2)
                n_border += 1
            elif depth_frac > 0.5 and particle.border > 8:
                if lon_frac - depth_frac < 0:
                    f_u = 2 - 2 * depth_frac
                    f_w = 4 * math.pow(1 - depth_frac, 2)
                    n_border += 1
                elif lon_frac - depth_frac > 0:
                    f_u = 4 * math.pow(1 - lon_frac, 2)
                    f_w = 2 - 2 * lon_frac
                    n_border += 1
            elif depth_frac > 0.5 and particle.border > 4:
                if lon_frac + depth_frac > 1:
                    f_u = 2-2 * depth_frac
                    f_w = 4 * math.pow(1 - depth_frac, 2)
                    n_border += 1
                elif lon_frac + depth_frac < 1:
                    f_u = 4 * math.pow(lon_frac, 2)
                    f_w = 2 * lon_frac
                    n_border += 1
        if B_binary == 1:  # border above
            if depth_frac < 0.5 and n_border == 0:
                f_u = 2 * depth_frac
                f_w = 4 * math.pow(depth_frac, 2)
                n_border += 1
            elif depth_frac < 0.5 and particle.border > 8:
                if lon_frac + depth_frac > 1:
                    f_u = 4 * math.pow(1 - lon_frac, 2)
                    f_w = 2-2 * lon_frac
                    n_border += 1
                elif lon_frac + depth_frac < 1:
                    f_u = 2 * depth_frac
                    f_w = 4 * math.pow(depth_frac, 2)
                    n_border += 1
            elif depth_frac < 0.5 and particle.border>4:
                if lon_frac - depth_frac < 0:
                    f_u = 4 * math.pow(lon_frac, 2)
                    f_w = 2 * lon_frac
                    n_border += 1
                elif lon_frac - depth_frac > 0:
                    f_u = 2 * depth_frac
                    f_w = 4 * math.pow(depth_frac, 2)
                    n_border += 1

    if fieldset.beaching == 3:  # linear - 0 towards wall velocities, linear - linear away
        if B_binary >= 8:  # border right
            B_binary += -8
            if lon_frac > 0.5 and u_final * particle.dt > 0:
                f_u = 0  # no cross-boundary movement
                f_w = 2 - 2 * lon_frac
                n_border += 1
            elif lon_frac > 0.5  and u_final * particle.dt <= 0:
                f_u = 2 - 2 * lon_frac
                f_w = 2 - 2 * lon_frac
                n_border += 1
        if B_binary >= 4:  # border left
            B_binary += -4
            if lon_frac < 0.5 and n_border == 0 and u_final * particle.dt < 0:
                f_u = 0
                f_w = 2 * lon_frac
                n_border += 1
            elif lon_frac < 0.5 and n_border == 0 and u_final * particle.dt >= 0:
                f_u = 2 * lon_frac
                f_w = 2 * lon_frac
                n_border += 1
        if B_binary >= 2:  # border below
            B_binary += -2
            if depth_frac > 0.5 and n_border == 0 and w_final * particle.dt > 0:
                f_u = 2 - 2 * depth_frac
                f_w = 0
                n_border += 1
            elif depth_frac > 0.5 and n_border == 0 and w_final * particle.dt <= 0:
                f_u = 2 - 2 * depth_frac
                f_w = 2 - 2 * depth_frac
                n_border += 1
            elif depth_frac > 0.5 and particle.border > 8:
                if lon_frac - depth_frac < 0 and w_final * particle.dt > 0:
                    f_u = 2 - 2 * depth_frac
                    f_w = 0
                    n_border += 1
                elif lon_frac - depth_frac < 0 and w_final * particle.dt <= 0:
                    f_u = 2 - 2 * depth_frac
                    f_w = 2 - 2 * depth_frac
                    n_border += 1
                elif lon_frac - depth_frac > 0 and u_final * particle.dt > 0:
                    f_u = 0
                    f_w = 2 - 2 * lon_frac
                    n_border += 1
                elif lon_frac - depth_frac > 0 and u_final * particle.dt <= 0:
                    f_u = 2 - 2 * lon_frac
                    f_w = 2 - 2 * lon_frac
                    n_border += 1
            elif depth_frac > 0.5 and particle.border > 4:
                if lon_frac + depth_frac > 1 and w_final * particle.dt > 0:
                    f_u = 2 - 2 * depth_frac
                    f_w = 0
                    n_border += 1
                elif lon_frac + depth_frac > 1 and w_final * particle.dt <= 0:
                    f_u = 2 - 2 * depth_frac
                    f_w = 2 - 2 * depth_frac
                    n_border += 1
                elif lon_frac + depth_frac < 1 and u_final * particle.dt < 0:
                    f_u = 0
                    f_w = 2 * lon_frac
                    n_border += 1
                elif lon_frac + depth_frac < 1 and u_final * particle.dt >= 0:
                    f_u = 2 * depth_frac
                    f_w = 2 * depth_frac
                    n_border += 1
        if B_binary == 1:  # border above
            if depth_frac < 0.5 and n_border == 0 and w_final * particle.dt < 0:
                f_u = 2 * depth_frac
                f_w = 0
                n_border += 1
            elif depth_frac < 0.5 and n_border == 0 and w_final * particle.dt >= 0:
                f_u = 2 * depth_frac
                f_w = 2 * depth_frac
                n_border += 1
            elif depth_frac < 0.5 and particle.border > 8:
                if lon_frac + depth_frac > 1 and u_final * particle.dt > 0:
                    f_u = 0
                    f_w = 2 - 2 * lon_frac
                    n_border += 1
                elif lon_frac + depth_frac > 1 and u_final * particle.dt <= 0:
                    f_u = 2 - 2 * lon_frac
                    f_w = 2 - 2 * lon_frac
                    n_border += 1
                elif lon_frac + depth_frac < 1 and w_final * particle.dt < 0:
                    f_u = 2 * depth_frac
                    f_w = 0
                    n_border += 1
                elif lon_frac + depth_frac < 1 and w_final * particle.dt >= 0:
                    f_u = 2 * depth_frac
                    f_w = 2 * depth_frac
                    n_border += 1
            elif depth_frac < 0.5 and particle.border > 4:
                if lon_frac - depth_frac < 0 and u_final * particle.dt < 0:
                    f_u = 0
                    f_w = 2 * lon_frac
                    n_border += 1
                elif lon_frac - depth_frac < 0 and u_final * particle.dt >= 0:
                    f_u = 2 * lon_frac
                    f_w = 2 * lon_frac
                    n_border += 1
                elif lon_frac - depth_frac > 0 and w_final * particle.dt < 0:
                    f_u = 2 * depth_frac
                    f_w = 0
                    n_border += 1
                elif lon_frac - depth_frac > 0 and w_final * particle.dt >= 0:
                    f_u = 2 * depth_frac
                    f_w = 2 * depth_frac
                    n_border += 1

    # if fieldset.beaching == 4: # linear - log all velocities
    #     f1 = f_linear1
    #     f2 = f_exp1
    #     if B_binary >= 8: # border right
    #         B_binary += -8
    #         if particle.d2c < fieldset.dx:
    #             f_lon = f2 # no cross-boundary movement
    #             f_depth = f1
    #             n_border += 1
    #     if B_binary >= 4: # border left
    #         B_binary += -4
    #         if particle.d2c < fieldset.dx and n_border == 0:
    #             f_lon = f2
    #             f_depth = f1
    #             n_border += 1
    #     if B_binary >= 2: # border below
    #         B_binary += -2
    #         if particle.d2c < fieldset.dx and n_border == 0:
    #             f_depth = f2
    #             f_lon = f1
    #             n_border += 1
    #         elif particle.d2c < fieldset.dx and particle.border > 8:
    #             if lon_frac-depth_frac < 0:
    #                 f_depth = f2
    #                 f_lon = f1
    #                 n_border += 1
    #             elif lon_frac-depth_frac > 0:
    #                 f_depth = f1
    #                 f_lon = f2
    #                 n_border += 1
    #         elif particle.d2c < fieldset.dx:
    #             if lon_frac+depth_frac > 1:
    #                 f_depth = f2
    #                 f_lon = f1
    #                 n_border += 1
    #             elif lon_frac+depth_frac < 1:
    #                 f_depth = f1
    #                 f_lon = f2
    #                 n_border += 1
    #     if B_binary == 1:  # border above
    #         if particle.d2c < fieldset.dx and n_border == 0:
    #             f_depth = f2
    #             f_lon = f1
    #             n_border += 1
    #         elif particle.d2c < fieldset.dx and particle.border > 8:
    #             if lon_frac + depth_frac > 1:
    #                 f_depth = f1
    #                 f_lon = f2
    #                 n_border += 1
    #             elif lon_frac + depth_frac < 1:
    #                 f_depth = f2
    #                 f_lon = f1
    #                 n_border += 1
    #         elif particle.d2c < fieldset.dx:
    #             if lon_frac - depth_frac < 0:
    #                 f_depth = f1
    #                 f_lon = f2
    #                 n_border += 1
    #             elif lon_frac - depth_frac > 0:
    #                 f_depth = f2
    #                 f_lon = f1
    #                 n_border += 1
    #
    # if fieldset.beaching == 5: # linear - sqrt
    #     f1 = f_linear1
    #     f2 = f_cub1
    #     if B_binary >= 8:  # border right
    #         B_binary += -8
    #         if particle.d2c < fieldset.dx:
    #             f_lon = f2  # no cross-boundary movement
    #             f_depth = f1
    #             n_border += 1
    #     if B_binary >= 4:  # border left
    #         B_binary += -4
    #         if particle.d2c < fieldset.dx and n_border == 0:
    #             f_lon = f2
    #             f_depth = f1
    #             n_border += 1
    #     if B_binary >= 2:  # border below
    #         B_binary += -2
    #         if particle.d2c < fieldset.dx and n_border == 0:
    #             f_depth = f2
    #             f_lon = f1
    #             n_border += 1
    #         elif particle.d2c < fieldset.dx and particle.border > 8:
    #             if lon_frac - depth_frac < 0:
    #                 f_depth = f2
    #                 f_lon = f1
    #                 n_border += 1
    #             elif lon_frac - depth_frac > 0:
    #                 f_depth = f1
    #                 f_lon = f2
    #                 n_border += 1
    #         elif particle.d2c < fieldset.dx:
    #             if lon_frac + depth_frac > 1:
    #                 f_depth = f2
    #                 f_lon = f1
    #                 n_border += 1
    #             elif lon_frac + depth_frac < 1:
    #                 f_depth = f1
    #                 f_lon = f2
    #                 n_border += 1
    #     if B_binary == 1:  # border above
    #         if particle.d2c < fieldset.dx and n_border == 0:
    #             f_depth = f2
    #             f_lon = f1
    #             n_border += 1
    #         elif particle.d2c < fieldset.dx and particle.border > 8:
    #             if lon_frac + depth_frac > 1:
    #                 f_depth = f1
    #                 f_lon = f2
    #                 n_border += 1
    #             elif lon_frac + depth_frac < 1:
    #                 f_depth = f2
    #                 f_lon = f1
    #                 n_border += 1
    #         elif particle.d2c < fieldset.dx:
    #             if lon_frac - depth_frac < 0:
    #                 f_depth = f1
    #                 f_lon = f2
    #                 n_border += 1
    #             elif lon_frac - depth_frac > 0:
    #                 f_depth = f2
    #                 f_lon = f1
    #                 n_border += 1

    if particle.closestobject == 22:
        f_u = 0
        f_w = 0

    particle.lon += f_u * u_final * particle.dt
    particle.lat += f_v * v_final * particle.dt
    particle.depth += f_w * w_final * particle.dt
    
def deleteparticle(particle,fieldset,time):
    """ This function deletes particles as they exit the domain and prints a message about their attributes at that moment
    """
    
    #print('Particle '+str(particle.id)+' has died at t = '+str(time)+' at lon, lat, depth = '+str(particle.lon)+', '+str(particle.lat)+', '+str(particle.depth))
    particle.delete()


def removeNaNs(particle,fieldset,time):
    """ This function removes the masked particles
    """
    if particle.lon == -999 and particle.depth==-999:
        particle.delete()

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

def coraldistancemap(edgemask,x_mesh,y_mesh):
    """ Function generating an array with the minimum distances to a coral object
    The edgemask object should be a boolean array with the same shape as the x- and y-meshes.
    It is True where the edges of the coral objects are located and False everywhere else
    """
    
    coralcoords = np.asarray((x_mesh.flatten(),y_mesh.flatten()))
    edgecoords = np.asarray((x_mesh[edgemask],y_mesh[edgemask]))
    D2coralmatrix = np.zeros((len(edgecoords[0]),len(coralcoords[0]))) # A matrix with the distance between each edge-coordinate and all coordinates
    
    for i in range(len(edgecoords[0])): # Loop over all edge-coordinates and calculate the distance to all points
        alldistances = np.sqrt(np.power(coralcoords[0,:]-edgecoords[0,i],2)+np.power(coralcoords[1,:]-edgecoords[1,i],2)) #distance from 1 edge-coordinate to all points
        D2coralmatrix[i, :] = alldistances
        coralmap = np.min(D2coralmatrix, axis=0)  # minimum distance of each point to all edge-coordinates
    

    coralmap = np.reshape(coralmap,x_mesh.shape)
    return coralmap

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

def CreateConmatrixSingle(d2cmax,dataset,objects,runtime):
    """Function creating a directed adjacency or connectivity matrix weighted by the amount of particles entering both respective boundary layers
    d2cmax is the depth of the boundary layer, dataset is an xarray dataset, objects contains Boolean arrays with the coorinates of each objects marked as True
    runtime is the runtime of the parcels execution"""

    connectivity = np.zeros((len(objects), len(objects)))

    for i in range(len(objects)): # loop through the starting objects
        # Find all trajectories that pass through the boundary layer
        a = dataset.where(np.logical_and(dataset['closestobject'].isin([i]), dataset['d2c'] < d2cmax))['trajectory'] # set to nan the trajectories that do not pass through the starting objects boundary layer
        a = np.nan_to_num(a, nan=-1) #
        a = np.unique(a) # all trajectory identifiers and -1, which is not an ID

        objectdata = dataset.where(dataset['trajectory'].isin([a]), drop=True) # make a subset of the trajectories
        objectdata = objectdata.where(objectdata['d2c'] < d2cmax) # set to nan all the data where the particle is not in a boundary layer

        for j in range(len(objects)):
            # count how many particles enter another boundary layer after they have entered the one of the starting object
            particles = 0
            for k in range(len(objectdata['traj'])): # go through each trajectory and see if they have an time instance (-> index) later in another boundary layer than the first time in the starting boundary layer
                if len(np.where(objectdata['closestobject'][k] == j)[0]) > 0: # particle has been in other boundary layer
                    if np.max(np.where(objectdata['closestobject'][k] == j)) > np.min(
                            np.where(objectdata['closestobject'][k] == i)): # particle appeared in other boundary layer AFTER having been in starting objects boundary layer
                        particles += 1
            connectivity[i, j] = particles / runtime # number of particles per second
        print(i)

    return connectivity

# Create a connectivity matrix with 'fresh' particles in # particles per second
def CreateConmatrixRepeat(d2cmax,dataset,objects,runtime):

    connectivity = np.zeros(len(objects))
    
    freshtraj = dataset.where(dataset['d2c'] < d2cmax, drop=True)['trajectory']
    freshtraj = np.nan_to_num(freshtraj, nan=-1)
    freshtraj = np.unique(freshtraj)
    freshdata = dataset.where(dataset['trajectory'].isin([freshtraj]), drop=True)
    freshvol = freshdata['pvolume'][:,0].values
    freshdata = freshdata.where(freshdata['d2c'] < d2cmax, drop=True)

    for i in range(len(freshdata['traj'])):
        objectnr = freshdata['closestobject'].isel(traj=i).dropna('obs')[0]
        if objectnr<len(objects):
            connectivity[int(objectnr)] += freshvol[i]/runtime
    return connectivity

# Create a connectivity matrix in # particles per meter per second
def CreateConmatrixSinglePerarea(d2cmax,dataset,objects,runtime,dx):
    
    connectivity = np.zeros((len(objects),len(objects)))
        
    for i in range(len(objects)):
        a = dataset.where(np.logical_and(dataset['closestobject'].isin([i]),dataset['d2c']<d2cmax))['trajectory']
        a = np.nan_to_num(a,nan=-1)
        a = np.unique(a)
        
        objectdata = dataset.where(dataset['trajectory'].isin([a]),drop=True)
        objectdata = objectdata.where(objectdata['d2c']<d2cmax)

        ob_xlen = np.count_nonzero(np.count_nonzero(objects[i],axis=0))
        ob_ylen = np.count_nonzero(np.count_nonzero(objects[i],axis=1))
        ob_x0 = np.nonzero(objects[i,-1])[0][1] - np.nonzero(objects[i,-1])[0][0] + 1

        objectarea = (2 * ob_ylen + 2*ob_xlen - ob_x0)*dx

        
        for j in range(len(objects)):
            particles = 0
            for k in range(len(objectdata['traj'])):
                if len(np.where(objectdata['closestobject'][k]==j)[0])>0:
                    if np.max(np.where(objectdata['closestobject'][k]==j))>np.min(np.where(objectdata['closestobject'][k] == i)):
                        particles += 1
            connectivity[i,j] = particles/objectarea/runtime
        print(i)
    
    return connectivity

# Create a connectivity matrix with 'fresh' particles in # particles per meter per second
def CreateConmatrixRepeatPerarea(d2cmax,dataset,objects,runtime,dx):

    connectivity = np.zeros(len(objects))

    objectarea = np.zeros(len(objects))
    for i in range(len(objects)):
        ob_xlen = np.count_nonzero(np.count_nonzero(objects[i], axis=0))
        ob_ylen = np.count_nonzero(np.count_nonzero(objects[i], axis=1))
        ob_x0 = np.nonzero(objects[i, -1])[0][1] - np.nonzero(objects[i, -1])[0][0] + 1

        objectarea[i] = (2 * ob_ylen + 2 * ob_xlen - ob_x0) * dx

    freshtraj = dataset.where(dataset['d2c'] < d2cmax, drop=True)['trajectory']
    freshtraj = np.nan_to_num(freshtraj, nan=-1)
    freshtraj = np.unique(freshtraj)
    freshdata = dataset.where(dataset['trajectory'].isin([freshtraj]), drop=True)
    freshvol = freshdata['pvolume'][:, 0].values
    freshdata = freshdata.where(freshdata['d2c'] < d2cmax, drop=True)

    for i in range(len(freshdata['traj'])):
        objectnr = freshdata['closestobject'].isel(traj=i).dropna('obs')[0]
        if objectnr<len(objects):
            connectivity[int(objectnr)] += freshvol[i]/objectarea[int(objectnr)]/runtime

    return connectivity

def TotalVolumeFlux(d2cmax,dataset,objects,runtime):
    TVF = np.zeros(len(objects))

    d2ctraj = dataset.where(dataset['d2c']<d2cmax, drop=True)['trajectory']
    d2ctraj = np.nan_to_num(d2ctraj, nan=-1)
    d2ctraj = np.unique(d2ctraj)
    d2cdata = dataset.where(dataset['trajectory'].isin([d2ctraj]), drop=True)
    for i in range(len(objects)):
        tvftraj = d2cdata.where(d2cdata['closestobject']==i,drop=True)['trajectory']
        tvftraj = np.nan_to_num(tvftraj, nan=-1)
        tvftraj = np.unique(tvftraj)
        tvfdata = dataset.where(dataset['trajectory'].isin([tvftraj]), drop=True)
        if len(tvfdata['traj'])>0:
            TVF[i] = np.sum(tvfdata['pvolume'][:,0].values)/runtime

    return TVF


def TotalVolumeFluxPerarea(d2cmax,dataset,objects,runtime,dx):
    TVF = np.zeros(len(objects))

    objectarea = np.zeros(len(objects))
    for i in range(len(objects)):
        ob_xlen = np.count_nonzero(np.count_nonzero(objects[i], axis=0))
        ob_ylen = np.count_nonzero(np.count_nonzero(objects[i], axis=1))
        ob_x0 = np.nonzero(objects[i, -1])[0][1] - np.nonzero(objects[i, -1])[0][0] + 1

        objectarea[i] = (2 * ob_ylen + 2 * ob_xlen - ob_x0) * dx

    d2ctraj = dataset.where(dataset['d2c']<d2cmax, drop=True)['trajectory']
    d2ctraj = np.nan_to_num(d2ctraj, nan=-1)
    d2ctraj = np.unique(d2ctraj)
    d2cdata = dataset.where(dataset['trajectory'].isin([d2ctraj]), drop=True)
    for i in range(len(objects)):
        tvftraj = d2cdata.where(d2cdata['closestobject']==i,drop=True)['trajectory']
        tvftraj = np.nan_to_num(tvftraj, nan=-1)
        tvftraj = np.unique(tvftraj)
        tvfdata = d2cdata.where(d2cdata['trajectory'].isin([tvftraj]), drop=True)
        if len(tvfdata['traj']) > 0:
            TVF[i] = np.sum(tvfdata['pvolume'][:, 0].values)/objectarea[i]/runtime

    return TVF

class DistParticle(JITParticle):  # Define a new particle class that contains three extra variables
    finaldistance = Variable('finaldistance', initial=0., dtype=np.float32)  # the distance travelled
    pvolume = Variable('pvolume', initial=0., dtype=np.float32)  # the volume each particle represents
    prevlon = Variable('prevlon', dtype=np.float32, to_write=False,
                        initial=attrgetter('lon'))  # the previous longitude
    prevlat = Variable('prevlat', dtype=np.float32, to_write=False,
                        initial=attrgetter('lat'))  # the previous latitude.
    prevdepth = Variable('prevdepth', dtype=np.float32, to_write=False,
                        initial=attrgetter('depth'))  # the previous latitude.
    d2c = Variable('d2c', dtype=np.float32, initial=0.)
    closestobject = Variable('closestobject', dtype=np.float32, initial=0.)
    border = Variable('border', dtype=np.float32, initial=0.)


def Samples(particle, fieldset, time):  # Custom function that samples d2c, closestobject and border at particle location
    particle.d2c = fieldset.D[time, particle.depth, particle.lat, particle.lon]
    particle.closestobject = fieldset.C[time, particle.depth, particle.lat, particle.lon]
    particle.border = fieldset.B[time, particle.depth, particle.lat, particle.lon]
    if particle.pvolume == 0. and particle.time:
        particle.pvolume = math.sqrt((fieldset.U[time, particle.depth, particle.lat, particle.lon])**2+(fieldset.W[time, particle.depth, particle.lat, particle.lon])**2)*fieldset.dx*fieldset.repeatdt