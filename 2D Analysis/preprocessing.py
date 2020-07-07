# -*- coding: utf-8 -*-
"""
Preprocessing python script to be executed before particle execution in my master thesis.

In this script the flowfields used as input are processed, mapping the following and saving them as pickle files:

    - The different objects/organisms in the flow
    - The distance to the nearest object
    - Which object is nearest
    - On which side of the cell a solid boundary exists

Output:
    - surfacemask
    - distancemask
    - objects
    - closestobject
    - bordermap

Created on Mon Feb 17 09:55:03 2020

@author: reint fischer
"""

import numpy as np
import xarray as xr
from functions import coraldistancemap, followPath

foldername = '16objects'
flow = 'uniform'

filename = flow+'.nc'
flowdata = xr.open_dataset(foldername+'/'+filename)
dx = np.abs(flowdata['X'][1]-flowdata['X'][0])
dy = dx

umask = np.ma.masked_invalid(flowdata['U'][0]) # masking the coral objects, where the velocity is NaN
xmesh,ymesh = np.meshgrid(np.arange(flowdata['X'].values[0]-0.5*dx, flowdata['X'].values[-1]+1.5*dx, dx),
                np.arange(flowdata['Y'].values[0]+0.5*dy, flowdata['Y'].values[-1]-1.5*dy, -dy))
x,y = np.meshgrid(flowdata['X'],flowdata['Y'])


# Coastmask
# Create a mask in which only the object cells bordering a fluid cell are masked
# 
# The coral objects are masked in the flow.
# By removing those masked cells that are surrounded on all sides by other masked cells
# the remaining mask contains only the object cells bordering the fluid.

surfacemask=np.copy(umask.mask)
for i in range(len(umask)):
    for j in range(len(umask[i])):
        if umask.mask[i,j] == True:
            if i>0 and i<len(umask)-1 and j>0 and j<len(umask[i])-1:
                if umask.mask[i-1,j]==True and umask.mask[i+1,j]==True and umask.mask[i,j-1]==True and umask.mask[i,j+1]==True:
                    surfacemask[i, j]=False
            elif i==0 and j>0 and j<len(umask[i])-1:
                if umask.mask[i+1,j]==True and umask.mask[i,j-1]==True and umask.mask[i,j+1]==True:
                    surfacemask[i, j]=False
            elif i==len(umask)-1 and j>0 and j<len(umask[i])-1:
                if umask.mask[i-1,j]==True and umask.mask[i,j-1]==True and umask.mask[i,j+1]==True:
                    surfacemask[i, j]=False
            elif i>0 and i<len(umask)-1 and j==0:
                if umask.mask[i-1,j]==True and umask.mask[i+1,j]==True and umask.mask[i,j+1]==True:
                    surfacemask[i, j]=False
            elif i>0 and i<len(umask)-1 and j==len(umask[i])-1:
                if umask.mask[i-1,j]==True and umask.mask[i+1,j]==True and umask.mask[i,j-1]==True:
                    surfacemask[i, j]=False
            elif i==0 and j==0:
                if umask.mask[i+1,j]==True and umask.mask[i,j+1]==True:
                    surfacemask[i, j]=False
            elif i==0 and j==len(umask[i])-1:
                if umask.mask[i+1,j]==True and umask.mask[i,j-1]==True:
                    surfacemask[i, j]=False
            elif i==len(umask)-1 and j==0:
                if umask.mask[i-1,j]==True and umask.mask[i,j+1]==True:
                    surfacemask[i, j]=False
            elif i==len(umask)-1 and j==len(umask[i])-1:
                if umask.mask[i-1,j]==True and umask.mask[i,j-1]==True:
                    surfacemask[i, j]=False
np.save(foldername +'/preprocessed/coastmask', surfacemask, allow_pickle=True)

# Distance map
#
# Create a mesh at each gridpoint with the values of the distance at that point to the nearest object.
# Compute the distance from the point to each edge point of the coral and take the smallest value.
# The edge points are provided by the coastmask created above.
distancemap = coraldistancemap(surfacemask, x, y)
np.save(foldername +'/preprocessed/distancemap', distancemap, allow_pickle=True)

# Objects
#
# Create numpy array of outer object gridcells of the shape (#objects,x,y)
# To analyse the influence of the flow on the different objects/organisms
# they need to be individually accessible
objects = np.zeros((0, len(surfacemask), len(surfacemask[0])), dtype=bool)
surfacemask = np.load(foldername +'/preprocessed/coastmask.npy')

check = 0
for i in range(len(surfacemask)):
    for j in range(len(surfacemask[i])):
        if surfacemask[i, j] == True:
            surfacemask[i, j] = False
            object1 = np.zeros(surfacemask.shape, dtype=bool)
            object1[i,j] = True
            surfacemask, object1, x1, y1 = followPath(i, j, surfacemask, object1)
            surfacemask, object1, x2, y2 = followPath(i, j, surfacemask, object1)

            check += 1
        if check>0:
            if -1<i+x1+1<len(surfacemask) and -1<j+y1<len(surfacemask[i]) and objects[:, i + x1 + 1, j + y1].any():
                objects[:][objects[:,i+x1+1,j+y1]] = np.ma.mask_or(objects[:][objects[:,i+x1+1,j+y1]],object1)
            elif -1<i+x1+1<len(surfacemask) and -1<j+y1+1<len(surfacemask[i]) and objects[:, i + x1 + 1, j + y1 + 1].any():
                objects[:][objects[:,i+x1+1,j+y1+1]] = np.ma.mask_or(objects[:][objects[:,i+x1+1,j+y1+1]],object1)
            elif -1<i+x1<len(surfacemask) and -1<j+y1+1<len(surfacemask[i]) and objects[:, i + x1, j + y1 + 1].any():
                objects[:][objects[:,i+x1,j+y1+1]] = np.ma.mask_or(objects[:][objects[:,i+x1,j+y1+1]],object1)
            elif -1<i+x1-1<len(surfacemask) and -1<j+y1+1<len(surfacemask[i]) and objects[:, i + x1 - 1, j + y1 + 1].any():
                objects[:][objects[:,i+x1-1,j+y1+1]] = np.ma.mask_or(objects[:][objects[:,i+x1-1,j+y1+1]],object1)
            elif -1<i+x1-1<len(surfacemask) and -1<j+y1<len(surfacemask[i]) and objects[:, i + x1 - 1, j + y1].any():
                objects[:][objects[:,i+x1-1,j+y1]] = np.ma.mask_or(objects[:][objects[:,i+x1-1,j+y1]],object1)
            elif -1<i+x1-1<len(surfacemask) and -1<j+y1-1<len(surfacemask[i]) and objects[:, i + x1 - 1, j + y1 - 1].any():
                objects[:][objects[:,i+x1-1,j+y1-1]] = np.ma.mask_or(objects[:][objects[:,i+x1-1,j+y1-1]],object1)
            elif -1<i+x1<len(surfacemask) and -1<j+y1-1<len(surfacemask[i]) and objects[:, i + x1, j + y1 - 1].any():
                objects[:][objects[:,i+x1,j+y1-1]] = np.ma.mask_or(objects[:][objects[:,i+x1,j+y1-1]],object1)
            elif -1<i+x1+1<len(surfacemask) and -1<j+y1-1<len(surfacemask[i]) and objects[:, i + x1 + 1, j + y1 - 1].any():
                objects[:][objects[:,i+x1+1,j+y1-1]] = np.ma.mask_or(objects[:][objects[:,i+x1+1,j+y1-1]],object1)
            elif -1<i+x2+1<len(surfacemask) and -1<j+y2<len(surfacemask[i]) and objects[:, i + x2 + 1, j + y2].any():
                objects[:][objects[:,i+x2+1,j+y2]] = np.ma.mask_or(objects[:][objects[:,i+x2+1,j+y2]],object1)
            elif -1<i+x2+1<len(surfacemask) and -1<j+y2+1<len(surfacemask[i]) and objects[:, i + x2 + 1, j + y2 + 1].any():
                objects[:][objects[:,i+x2+1,j+y2+1]] = np.ma.mask_or(objects[:][objects[:,i+x2+1,j+y2+1]],object1)
            elif -1<i+x2<len(surfacemask) and -1<j+y2+1<len(surfacemask[i]) and objects[:, i + x2, j + y2 + 1].any():
                objects[:][objects[:,i+x2,j+y2+1]] = np.ma.mask_or(objects[:][objects[:,i+x2,j+y2+1]],object1)
            elif -1<i+x2-1<len(surfacemask) and -1<j+y2+1<len(surfacemask[i]) and objects[:, i + x2 - 1, j + y2 + 1].any():
                objects[:][objects[:,i+x2-1,j+y2+1]] = np.ma.mask_or(objects[:][objects[:,i+x2-1,j+y2+1]],object1)
            elif -1<i+x2-1<len(surfacemask) and -1<j+y2<len(surfacemask[i]) and objects[:, i + x2 - 1, j + y2].any():
                objects[:][objects[:,i+x2-1,j+y2]] = np.ma.mask_or(objects[:][objects[:,i+x2-1,j+y2]],object1)
            elif -1<i+x2-1<len(surfacemask) and -1<j+y2-1<len(surfacemask[i]) and objects[:, i + x2 - 1, j + y2 - 1].any():
                objects[:][objects[:,i+x2-1,j+y2-1]] = np.ma.mask_or(objects[:][objects[:,i+x2-1,j+y2-1]],object1)
            elif -1<i+x2<len(surfacemask) and -1<j+y2-1<len(surfacemask[i]) and objects[:, i + x2, j + y2 - 1].any():
                objects[:][objects[:,i+x2,j+y2-1]] = np.ma.mask_or(objects[:][objects[:,i+x2,j+y2-1]],object1)
            elif -1<i+x2+1<len(surfacemask) and -1<j+y2-1<len(surfacemask[i]) and objects[:, i + x2 + 1, j + y2 - 1].any():
                objects[:][objects[:,i+x2+1,j+y2-1]] = np.ma.mask_or(objects[:][objects[:,i+x2+1,j+y2-1]],object1)
            else:
                object1 = np.expand_dims(object1,axis=0)
                objects = np.concatenate((objects,object1),axis=0)
            check = 0

# Sort objects for x-component
idx = []
for i in range(len(objects)):
    idx += [np.mean(x[objects[i]])]
objects = objects[np.argsort(idx)]
np.save(foldername+'/preprocessed/'+'objects',objects,allow_pickle=True)

# Create closest object map
distancearray = np.zeros((len(objects), len(objects[0]), len(objects[0, 0]))) # Initialise object containing the distances to each object
for i in range(len(objects)):
    distancearray[i] = coraldistancemap(objects[i], x, y)
closestobject = np.argmin(distancearray, axis=0)
closestobject[umask.mask] = [len(objects)]*np.sum(umask.mask)
np.save(foldername +'/preprocessed/' +'closestobject', closestobject, allow_pickle=True)

# Create the coastal map
surfacemask = np.load(foldername+'/preprocessed/coastmask.npy')
bordermap = np.zeros(x.shape)
for i in range(len(x)):
    for j in range(len(x[i])):
        if i > 0 and i < len(x) - 1 and j > 0 and j < len(x[i]) - 1:
            if surfacemask[i - 1, j] == True:
                bordermap[i,j] += 1
            if surfacemask[i + 1, j] == True:
                bordermap[i,j] += 2
            if surfacemask[i, j - 1] == True:
                bordermap[i,j] += 4
            if surfacemask[i, j + 1] == True:
                bordermap[i, j] += 8
        elif i == 0 and j > 0 and j < len(x[i]) - 1:
            if surfacemask[i + 1, j] == True:
                bordermap[i,j] += 2
            if surfacemask[i, j - 1] == True:
                bordermap[i,j] += 4
            if surfacemask[i, j + 1] == True:
                bordermap[i,j] += 8
        elif i == len(x) - 1 and j > 0 and j < len(x[i]) - 1:
            if surfacemask[i - 1, j] == True:
                bordermap[i,j] += 1
            if surfacemask[i, j - 1] == True:
                bordermap[i,j] += 4
            if surfacemask[i, j + 1] == True:
                bordermap[i,j] += 8
        elif i > 0 and i < len(x) - 1 and j == 0:
            if surfacemask[i - 1, j] == True:
                bordermap[i,j] += 1
            if surfacemask[i + 1, j] == True:
                bordermap[i,j] += 2
            if surfacemask[i, j + 1] == True:
                bordermap[i,j] += 8
        elif i > 0 and i < len(x) - 1 and j == len(x[i]) - 1:
            if surfacemask[i - 1, j] == True:
                bordermap[i,j] += 1
            if surfacemask[i + 1, j] == True:
                bordermap[i,j] += 2
            if surfacemask[i, j - 1] == True:
                bordermap[i, j] += 4
        elif i == 0 and j == 0:
            if surfacemask[i + 1, j] == True:
                bordermap[i,j] += 2
            if surfacemask[i, j + 1] == True:
                bordermap[i, j] += 8
        elif i == 0 and j == len(x[i]) - 1:
            if surfacemask[i + 1, j] == True:
                bordermap[i,j] += 2
            if umask.mask[i, j - 1] == True:
                bordermap[i, j] += 4
        elif i == len(umask) - 1 and j == 0:
            if surfacemask[i - 1, j] == True:
                bordermap += 1
            if surfacemask[i, j + 1] == True:
                bordermap[i, j] += 8
        elif i == len(x) - 1 and j == len(x[i]) - 1:
            if surfacemask[i - 1, j] == True:
                bordermap[i,j] += 1
            if surfacemask[i, j - 1] == True:
                bordermap[i, j] += 4
np.save(foldername+'/preprocessed/'+'bordermap',bordermap,allow_pickle=True)
