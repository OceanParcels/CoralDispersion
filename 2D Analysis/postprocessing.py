# -*- coding: utf-8 -*-
"""
Postprocessing python script to be executed after particle execution in my master thesis.

In this script the particle output from parcels is processed and connectivity matrices are calculated
The differences between matrices are due to the presence of 'fresh' particles and whether or not
the edge strength is weighted by the area of the object receiving the particles

Output:
    - connectivity matrix

How to use:
Define the parameters of the particleset you want to analyse to use as input for the run function in the bottom of the script.
Then run the python file.

Created on Mon May 18 23:55:03 2020

@author: reint fischer
"""

import numpy as np
import xarray as xr
from functions import AdjacencyMatrix, VolumeFluxes, AdjacencyMatrixPerarea, VolumeFluxesPerarea, TotalVolumeFlux, TotalVolumeFluxPerarea
from datetime import timedelta

def run(flow,dt,bconstant,repeat,repeatdt = 0.1,foldername ='21objects',d2cmax=0.05):
    tstep = str(dt)[2:]
    beaching_strategy = str(bconstant)
    flow = flow
    fb = 'forward'  # variable to determine whether the flowfields are analysed 'forward' or 'backward' in time

    labellist = [0, 1, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 19, 21]
    notlist = [2,5,8,14,18,20]

    if repeat:
        dfilename = 'r'+str(repeatdt)[2:]+'-B' + beaching_strategy + '-' + flow + '-' + tstep + '-' + fb + '.nc'
    else:
        dfilename = 'B' + beaching_strategy + '-' + flow + '-' + tstep + '-' + fb + '.nc'

    data = xr.open_dataset(foldername + '/pfiles/' + dfilename)

    dx = data.attrs['dx']

    runtimelist = data.attrs['runtime'].split()
    runtime = timedelta(seconds=float(runtimelist[0]))

    objects = np.load(foldername + '/preprocessed/' + 'objects.npy')

    if repeat:
        conmatrix = VolumeFluxes(d2cmax, data, objects, runtime.total_seconds())
        np.save(
            foldername + '/postprocessed/freshmatrix-Repeat-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                d2cmax), conmatrix, allow_pickle=True)
        conmatrix = VolumeFluxesPerarea(d2cmax, data, objects, runtime.total_seconds(), dx)
        np.save(
            foldername + '/postprocessed/freshmatrix-Repeat-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                d2cmax), conmatrix, allow_pickle=True)
        TVF = TotalVolumeFlux(d2cmax, data, objects, runtime.total_seconds())
        np.save(
            foldername + '/postprocessed/totalvolumeflux-Repeat-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                d2cmax), TVF, allow_pickle=True)
        TVF = TotalVolumeFluxPerarea(d2cmax, data, objects, runtime.total_seconds(), dx)
        np.save(
            foldername + '/postprocessed/totalvolumeflux-Repeat-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                d2cmax), TVF, allow_pickle=True)
        if len(objects)>20:
            conmatrix = np.load(
                foldername + '/postprocessed/freshmatrix-Repeat-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax) + '.npy')
            conmatrix = conmatrix[labellist]
            np.save(
                foldername + '/postprocessed/freshmatrix-Repeat-Partial-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax), conmatrix, allow_pickle=True)
            conmatrix = np.load(
                foldername + '/postprocessed/freshmatrix-Repeat-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax) + '.npy')
            conmatrix = conmatrix[labellist]
            np.save(
                foldername + '/postprocessed/freshmatrix-Repeat-Partial-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax), conmatrix, allow_pickle=True)
            TVF = np.load(
                foldername + '/postprocessed/totalvolumeflux-Repeat-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax) + '.npy')
            TVF = TVF[labellist]
            np.save(
                foldername + '/postprocessed/totalvolumeflux-Repeat-Partial-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax), TVF, allow_pickle=True)
            TVF = np.load(
                foldername + '/postprocessed/totalvolumeflux-Repeat-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax) + '.npy')
            TVF = TVF[labellist]
            np.save(
                foldername + '/postprocessed/totalvolumeflux-Repeat-Partial-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax), TVF, allow_pickle=True)
        else:
            conmatrix = np.load(
                foldername + '/postprocessed/freshmatrix-Repeat-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax) + '.npy')
            for i in notlist:
                conmatrix = np.insert(conmatrix, i, 0)
            np.save(
                foldername + '/postprocessed/freshmatrix-Repeat-Sparse-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax), conmatrix, allow_pickle=True)
            conmatrix = np.load(
                foldername + '/postprocessed/freshmatrix-Repeat-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax) + '.npy')
            for i in notlist:
                conmatrix = np.insert(conmatrix, i, 0)
            np.save(
                foldername + '/postprocessed/freshmatrix-Repeat-Sparse-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax), conmatrix, allow_pickle=True)
            TVF = np.load(
                foldername + '/postprocessed/totalvolumeflux-Repeat-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax) + '.npy')
            for i in notlist:
                TVF = np.insert(TVF, i, 0)
            np.save(
                foldername + '/postprocessed/totalvolumeflux-Repeat-Sparse-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax), TVF, allow_pickle=True)
            TVF = np.load(
                foldername + '/postprocessed/totalvolumeflux-Repeat-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax) + '.npy')
            for i in notlist:
                TVF = np.insert(TVF, i, 0)
            np.save(
                foldername + '/postprocessed/totalvolumeflux-Repeat-Sparse-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax), TVF, allow_pickle=True)
    else:
        conmatrix = AdjacencyMatrix(d2cmax, data, objects, runtime.total_seconds())
        np.save(foldername + '/postprocessed/conmatrix-Single-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                d2cmax), conmatrix, allow_pickle=True)
        conmatrix = AdjacencyMatrixPerarea(d2cmax, data, objects, runtime.total_seconds(), dx)
        np.save(foldername + '/postprocessed/conmatrix-Single-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                d2cmax), conmatrix, allow_pickle=True)
        if len(objects)>20:
            conmatrix = np.load(foldername + '/postprocessed/conmatrix-Single-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                d2cmax)+'.npy')
            conmatrix = conmatrix[labellist][:, labellist]
            np.save(
                foldername + '/postprocessed/conmatrix-Single-Partial-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax), conmatrix, allow_pickle=True)
            conmatrix = np.load(
                foldername + '/postprocessed/conmatrix-Single-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax) + '.npy')
            conmatrix = conmatrix[labellist][:, labellist]
            np.save(
                foldername + '/postprocessed/conmatrix-Single-Partial-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax), conmatrix, allow_pickle=True)
        else:
            conmatrix = np.load(
                foldername + '/postprocessed/conmatrix-Single-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax) + '.npy')
            for i in notlist:
                conmatrix = np.insert(conmatrix, i, 0, axis=0)
            for j in notlist:
                conmatrix = np.insert(conmatrix, j, 0, axis=1)
            np.save(
                foldername + '/postprocessed/conmatrix-Single-Sparse-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax), conmatrix, allow_pickle=True)
            conmatrix = np.load(
                foldername + '/postprocessed/conmatrix-Single-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax) + '.npy')
            for i in notlist:
                conmatrix = np.insert(conmatrix, i, 0, axis=0)
            for j in notlist:
                conmatrix = np.insert(conmatrix, j, 0, axis=1)
            np.save(
                foldername + '/postprocessed/conmatrix-Single-Sparse-Perarea-' + beaching_strategy + '-' + flow + '-' + tstep + '-' + str(
                    d2cmax), conmatrix, allow_pickle=True)



if __name__ == "__main__":
    run('waveparabolic', 0.001, 2, True, foldername='21objects')
    run('parabolic', 0.001, 2, True, foldername='21objects')
