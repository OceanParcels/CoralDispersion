import numpy as np
import xarray as xr
from datetime import timedelta

import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmocean
import seaborn as sns
import random


def run(foldername='21objects',tstep='001',beaching_strategy='2', flow = 'waveparabolic', d2cmax=0.05, repeatdt=0.1):
    fb = 'forward'
    filename = flow + '.nc'  # Flowdata used as input for the particle simulation
    dfilename = 'spinr' + str(repeatdt)[2:] + '-B' + beaching_strategy + '-' + flow + '-' + tstep + '-' + fb + '.nc'

    flowdata = xr.open_dataset(foldername + '/' + filename)
    data = xr.open_dataset(foldername + '/pfiles/' + dfilename)

    dx = data.attrs['dx']  # Spatial resolution
    dy = dx  # If gridcells are square, dx = dy

    outputdtlist = data.attrs['outputdt'].split()  # Timestep at which particledata are written
    outputdt = timedelta(seconds=float(outputdtlist[0]))

    dtlist = data.attrs['dt'].split()  # Timestep at which new locations are calculated
    dt = timedelta(seconds=float(dtlist[0]))

    runtimelist = data.attrs['runtime'].split()  # Total runtime of the simulation
    runtime = timedelta(seconds=float(runtimelist[0]))

    coralfield = np.zeros(flowdata['U'][0, :, :].shape)  # Flowdata at t = 0, with solid objects as NaN
    coralmesh = np.ma.masked_array(coralfield, ~np.ma.masked_invalid(
        flowdata['U'][1, :, :]).mask)  # Array masking all points of the mesh except the solid objects
    x, y = np.meshgrid(flowdata['X'], flowdata['Y'])  # Meshgrid of x and y coordinates that can be masked to draw objects
    xmesh, ymesh = np.meshgrid(np.arange(flowdata['X'].values[0] - 0.5 * dx, flowdata['X'].values[-1] + 0.5 * dx, dx),
                               np.arange(flowdata['Y'].values[0] + 0.5 * dy, flowdata['Y'].values[-1] - 1.5 * dy,
                                         -dy))  # Meshgrid of x and y coordinates staggered with 0.5*dx and 0.5*dy to draw squares at x and y with pcolormesh
    objects = np.load(
        foldername + '/preprocessed/' + 'objects.npy')  # The separated masks of all objects as defined in preprocessing
    bounds = np.linspace(-0.5, len(objects) + 0.5,
                         len(objects) + 2)  # The bounds for the colormapping of a qualitative color to each object
    labellist = [0, 1, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 19,
                 21]  # The numbers of the objects that remain in the run with certain objects left out
    oblist = [0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14, 14, 15, 15]

    plottimes = np.arange(np.nanmin(data['time'].values), np.nanmax(data['time'].values),
                          outputdt)  # , dtype='datetime64[ns]')

    # Animation of a particleset with particles initialised at the leftside boundary of the domain
    # Particles that end up entering boundary layers with a thickness of d2cmax are color-coded with the object they reach first.
    # `data` is the variable name of the xarray dataset with all the particle information

    # Find all the objects that are reached so we can color-code them and store the object numbers in `reached`
    freshdata = data.where(data['d2c'] < d2cmax,drop=True)  # only particles that enter boundary layers
    freshdata = freshdata.where(freshdata['closestobject']<len(objects),drop =True)
    reached = []
    for i in range(len(freshdata['traj'])):
        reached += [int(freshdata['closestobject'].isel(traj=i).dropna('obs')[0])]  # the first object that is reached
    reached = np.array(reached)  # first object number for each trajectory
    if len(objects) < 20:
        reached = np.array(labellist)[reached]

    # `edata` is a subset of data with the entire trajectories of the particles that enter boundary layers.
    # the amount of trajectories is the same as the length of `reached`
    e = data.where(data['d2c'] < d2cmax, drop=True)
    e = e.where(e['closestobject']<len(objects), drop=True)['trajectory']
    e = np.nan_to_num(e, nan=-1)
    e = np.unique(e)
    edata = data.where(data['trajectory'].isin([e]), drop=True)

    # Create a color palette for the reached objects and corresponding particles

    freshpalette = sns.hls_palette(22, l=.7, s=.8)  # the palette of colors that is used to color the objects
    random.seed(2)
    random.shuffle(freshpalette)  # randomise the colors so that objects close together are clearly distinguishable
    fresh_cmap = ListedColormap(
        freshpalette)  # the scatter plots needs a colormap to color the points based on their values in `reached`
    freshbounds = np.arange(
        23) - 0.5  # the boundaries of the values in `reached` on which the ListedColormap is projected. Len = len(fresh_cmap)+1

    fig = plt.figure(figsize=(18, 2))  # Initialise an elongated figure to keep correct proportions of the flowfield
    widths = [1.4, 9]  # Define the relative widths of the barplot and particle animation
    spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=widths,
                            wspace=0.01)  # 2 columns, 1 for the bar and 1 for the particles

    # `ax0` is the name of the axis on which the barplot animation is drawn
    ax0 = fig.add_subplot(spec[0])
    d = np.linspace(-0.5, 0.5, 11)  # Boundaries at which to draw the bars
    zinits = np.zeros(len(d) - 1)  # Initialise the bar lengths
    ax0.set_xlim(0., 0.8)
    ax0.invert_xaxis()
    ax0.set_yticks(d[::2])  # Label every second bar
    ax0.set_ylabel("Entering height (m)")
    ax0.set_xlabel("Cumulative fraction of released particles")
    ax0.set_ylim(-0.5, 0.5)

    # `ax1` is the name of the variable where the particle animation is drawn
    ax1 = fig.add_subplot(spec[1])
    ax1.set_facecolor('#d6fffe')  # lightblue background
    ax1.tick_params(  # Set the height ticks and labels on the right instead of left
        which='both',
        left=False,
        labelleft=False,
        right=True,
        labelright=True)
    pc = ax1.pcolormesh(xmesh, ymesh, coralmesh, cmap=cmocean.cm.gray)  # draw the coral objects in a gray/black

    etime = np.where(np.logical_and(edata['time'] >= plottimes[0],
                                    edata['time'] < plottimes[1]))  # selection of edata to draw at initial timestep
    dtime = np.where(np.logical_and(data['time'] >= plottimes[0],
                                    data['time'] < plottimes[1]))  # selection of other particles to draw in grey

    sc2 = ax1.scatter(data['lon'].values[dtime], -data['z'].values[dtime], c='lightgray', s=10, marker="o",
                      alpha=0.2)  # draw initial grey particles
    sc = ax1.scatter(edata['lon'].values[etime], -edata['z'].values[etime], c=reached[etime[0]],
                     norm=mpl.colors.BoundaryNorm(freshbounds, fresh_cmap.N), s=10, marker="o", cmap=fresh_cmap)

    # calculate the barlengths to draw initially in `zinits`
    z = np.where(np.logical_and(np.logical_and(edata['time'] >= plottimes[0], edata['time'] < plottimes[1]),
                                edata['lon'] == flowdata['X'][0].values))  # Particles are initialised at flowdata['X'][0]
    zs, bins = np.histogram(-edata['z'].values[z], d)  # bin the initialised particles in the bins defined for `ax0` above
    zinits = np.add(zinits, zs)  # add the initialised particles to the empty bar-lengths
    barh = ax0.barh(d[:-1], zinits / np.sum(zinits), 0.1,
                    align='edge')  # draw the horizontal bars with the edge at the bottom boundary of the bin
    zarray = np.zeros((len(zs), len(data['obs']) - 2))  # the array of incoming particle heights at each timestep
    zarray[:, 0] = zs

    time_text = ax1.text(-1.85, 0.44, '', horizontalalignment='left', verticalalignment='top')  # Initialise time ticker

    # Number the reached objects in a color from the palette
    for i in range(22):
        if i in np.unique(reached):  # Only number the relevant objects
            if len(objects) > 20:
                ax1.text(np.mean(x[objects[i]]), np.mean(y[objects[i]]) - 0.01, str(i), horizontalalignment='center',
                         verticalalignment='center', color=freshpalette[i], weight='bold')
            else:
                ax1.text(np.mean(x[objects[oblist[i]]]), np.mean(y[objects[oblist[i]]]) - 0.01, str(i),
                         horizontalalignment='center', verticalalignment='center', color=freshpalette[i], weight='bold')


    def animate(i, sc, sc2, zinits, zarray):
        # remove the existing bars to be able to redraw them without clearing the other settings for `ax0`
        for bar in ax0.containers:
            bar.remove()
        # calculate the additional length of the bars
        z = np.where(np.logical_and(np.logical_and(edata['time'] >= plottimes[i], edata['time'] < plottimes[i + 1]),
                                    edata['lon'] == -0.4921875))
        zs, bins = np.histogram(-edata['z'].values[z], d)
        zinits += zs  # add the amount of particles
        zarray[:, i] = zs
        ax0.barh(d[:-1], zinits / np.sum(zinits), 0.1, align='edge', color='k')  # redraw the bars

        # Redraw the particle positions at the correct time
        etime = np.where(np.logical_and(edata['time'] >= plottimes[i], edata['time'] < plottimes[i + 1]))
        dtime = np.where(np.logical_and(data['time'] >= plottimes[i], data['time'] < plottimes[i + 1]))
        sc2.set_offsets(np.c_[data['lon'].values[dtime], -data['z'].values[dtime]])  # grey particles
        sc.set_offsets(np.c_[edata['lon'].values[etime], -edata['z'].values[etime]])  # colored particles
        sc.set_array(reached[etime[
            0]])  # set the correct color for the colored particles, important for the newly initialised particles
        ts = i * outputdt.total_seconds()  # Calculate the time
        time_text.set_text('time = %.1f seconds' % ts)  # Update the time
        return sc, sc2, zinits, zarray


    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("height (m)")
    ax1.yaxis.set_label_position("right")
    ax1.set_xlim(-0.5, 8.5)
    ax1.set_ylim(-0.5, 0.5)
    anim = animation.FuncAnimation(fig, animate, fargs=(sc, sc2, zinits, zarray),
                                   frames=len(data['lon'][0]) - 2)
    plt.subplots_adjust(bottom=0.22)  # Adjust the location of the bottom of the drawn axes so the x-labels are visible
    anim.save('Figures/animation' + foldername + 'spinfresh-B' + beaching_strategy + '-' + flow + '-' + fb + '-' + str(
        runtime.seconds) + '-' + str(d2cmax)[2:] + '.mp4')
    plt.close(fig)

    fig = plt.figure(figsize=(18, 2))  # Initialise an elongated figure to keep correct proportions of the flowfield
    widths = [1.4, 9]  # Define the relative widths of the barplot and particle animation
    spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=widths,
                            wspace=0.01)  # 2 columns, 1 for the bar and 1 for the particles

    # `ax0` is the name of the axis on which the barplot animation is drawn
    ax0 = fig.add_subplot(spec[0])
    ax0.set_xlim(0., 0.8)
    ax0.invert_xaxis()
    ax0.set_yticks(d[::2])  # Label every second bar
    ax0.set_ylabel("Entering height (m)")
    ax0.set_xlabel("Cumulative fraction of released particles")
    ax0.set_ylim(-0.5, 0.5)

    ax0.barh(d[:-1], zinits / np.sum(zinits), 0.1, align='edge', color='k')

    # `ax1` is the name of the variable where the particle animation is drawn
    ax1 = fig.add_subplot(spec[1])
    ax1.set_facecolor('#d6fffe')  # lightblue background
    ax1.tick_params(  # Set the height ticks and labels on the right instead of left
        which='both',
        left=False,
        labelleft=False,
        right=True,
        labelright=True)

    # traj2 = ax1.plot(data['lon'].values.transpose(), -data['z'].values.transpose(), c='lightgray', alpha=0.2) # draw initial grey particles
    f, indices = np.unique(reached, return_inverse=True)
    for i in range(22):
        if i in np.unique(reached):
            j = np.argwhere(np.unique(reached) == i)
            obindices = np.argwhere(indices == j)
            fdata = edata.where(edata['traj'].isin(obindices))
            ax1.plot(fdata['lon'].values.transpose(), -fdata['z'].values.transpose(), c=freshpalette[i], alpha=0.05,
                     linewidth=1,zorder=1)

    ax1.pcolormesh(xmesh, ymesh, coralmesh, cmap=cmocean.cm.gray,zorder=2)  # draw the coral objects in a gray/black
    # Number the reached objects in a color from the palette
    for i in range(22):
        if i in np.unique(reached):  # Only number the relevant objects
            if len(objects) > 20:
                ax1.text(np.mean(x[objects[i]]), np.mean(y[objects[i]]) - 0.01, str(i), horizontalalignment='center',
                         verticalalignment='center', color=freshpalette[i], weight='bold',zorder=3)
            else:
                ax1.text(np.mean(x[objects[oblist[i]]]), np.mean(y[objects[oblist[i]]]) - 0.01, str(i),
                         horizontalalignment='center', verticalalignment='center', color=freshpalette[i], weight='bold',zorder=3)

    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("height (m)")
    ax1.yaxis.set_label_position("right")
    ax1.set_xlim(-0.5, 8.5)
    ax1.set_ylim(-0.5, 0.5)
    plt.subplots_adjust(bottom=0.22)  # Adjust the location of the bottom of the drawn axes so the x-labels are visible

    plt.savefig('Figures/trajectories' + foldername + 'spinfresh-B' + beaching_strategy + '-' + flow + '-' + fb + '-' + str(
        runtime.seconds) + '-' + str(d2cmax)[2:])
    plt.close(fig)

    np.save(foldername + '/postprocessed/spinzinits' + beaching_strategy + '-' + flow + '-' + fb + '-' + str(
        runtime.seconds) + '-' + str(d2cmax)[2:], zinits, allow_pickle=True)

if __name__ == "__main__":
    run('21objects', '001', '0', 'waveparabolic',d2cmax=0.05)
    run('21objects', '001', '0', 'parabolic',d2cmax=0.05)
    run('16objects', '001', '0', 'waveparabolic',d2cmax=0.05)
    run('16objects', '001', '0', 'parabolic',d2cmax=0.05)
    run('21objects', '001', '0', 'waveparabolic', d2cmax=0.04)
    run('21objects', '001', '0', 'parabolic', d2cmax=0.04)
    run('16objects', '001', '0', 'waveparabolic', d2cmax=0.04)
    run('16objects', '001', '0', 'parabolic', d2cmax=0.04)
    run('21objects', '001', '0', 'waveparabolic', d2cmax=0.03)
    run('21objects', '001', '0', 'parabolic', d2cmax=0.03)
    run('16objects', '001', '0', 'waveparabolic', d2cmax=0.03)
    run('16objects', '001', '0', 'parabolic', d2cmax=0.03)
    run('21objects', '001', '0', 'waveparabolic', d2cmax=0.02)
    run('21objects', '001', '0', 'parabolic', d2cmax=0.02)
    run('16objects', '001', '0', 'waveparabolic', d2cmax=0.02)
    run('16objects', '001', '0', 'parabolic', d2cmax=0.02)
    run('21objects', '001', '1', 'waveparabolic', d2cmax=0.05)
    run('21objects', '001', '1', 'parabolic', d2cmax=0.05)
    run('16objects', '001', '1', 'waveparabolic', d2cmax=0.05)
    run('16objects', '001', '1', 'parabolic', d2cmax=0.05)
    run('21objects', '001', '1', 'waveparabolic', d2cmax=0.04)
    run('21objects', '001', '1', 'parabolic', d2cmax=0.04)
    run('16objects', '001', '1', 'waveparabolic', d2cmax=0.04)
    run('16objects', '001', '1', 'parabolic', d2cmax=0.04)
    run('21objects', '001', '1', 'waveparabolic', d2cmax=0.03)
    run('21objects', '001', '1', 'parabolic', d2cmax=0.03)
    run('16objects', '001', '1', 'waveparabolic', d2cmax=0.03)
    run('16objects', '001', '1', 'parabolic', d2cmax=0.03)
    run('21objects', '001', '1', 'waveparabolic', d2cmax=0.02)
    run('21objects', '001', '1', 'parabolic', d2cmax=0.02)
    run('16objects', '001', '1', 'waveparabolic', d2cmax=0.02)
    run('16objects', '001', '1', 'parabolic', d2cmax=0.02)
    run('21objects', '001', '3', 'waveparabolic', d2cmax=0.05)
    run('21objects', '001', '3', 'parabolic', d2cmax=0.05)
    run('16objects', '001', '3', 'waveparabolic', d2cmax=0.05)
    run('16objects', '001', '3', 'parabolic', d2cmax=0.05)
    run('21objects', '001', '3', 'waveparabolic', d2cmax=0.04)
    run('21objects', '001', '3', 'parabolic', d2cmax=0.04)
    run('16objects', '001', '3', 'waveparabolic', d2cmax=0.04)
    run('16objects', '001', '3', 'parabolic', d2cmax=0.04)
    run('21objects', '001', '3', 'waveparabolic', d2cmax=0.03)
    run('21objects', '001', '3', 'parabolic', d2cmax=0.03)
    run('16objects', '001', '3', 'waveparabolic', d2cmax=0.03)
    run('16objects', '001', '3', 'parabolic', d2cmax=0.03)
    run('21objects', '001', '3', 'waveparabolic', d2cmax=0.02)
    run('21objects', '001', '3', 'parabolic', d2cmax=0.02)
    run('16objects', '001', '3', 'waveparabolic', d2cmax=0.02)
    run('16objects', '001', '3', 'parabolic', d2cmax=0.02)
    # run('waveparabolic', 0.001, 1, foldername='21objects')
    # run('parabolic', 0.001, 1, foldername='21objects')
    # run('waveparabolic', 0.001, 1, foldername='16objects')
    # run('parabolic', 0.001, 1, foldername='16objects')
    # run('waveparabolic', 0.001, 3, foldername='21objects')
    # run('parabolic', 0.001, 3, foldername='21objects')
    # run('waveparabolic', 0.001, 3, foldername='16objects')
    # run('parabolic', 0.001, 3, foldername='16objects')
    # run(foldername='16objects',beaching_strategy='1', flow='parabolic')
    # run(foldername='16objects',beaching_strategy='1')
    # run(beaching_strategy='1', flow='parabolic')
    # run(beaching_strategy='1')
    # run(foldername='16objects', beaching_strategy='3', flow='parabolic')
    # run(foldername='16objects', beaching_strategy='3')
    # run(beaching_strategy='3', flow='parabolic')
    # run(beaching_strategy='3')
    # run(foldername='16objects', beaching_strategy='1', flow='parabolic',d2cmax=0.04)
    # run(foldername='16objects', beaching_strategy='1',d2cmax=0.04)
    # run(beaching_strategy='1', flow='parabolic',d2cmax=0.04)
    # run(beaching_strategy='1',d2cmax=0.04)
    # run(foldername='16objects', beaching_strategy='3', flow='parabolic',d2cmax=0.04)
    # run(foldername='16objects', beaching_strategy='3',d2cmax=0.04)
    # run(beaching_strategy='3', flow='parabolic',d2cmax=0.04)
    # run(beaching_strategy='3',d2cmax=0.04)
    # run(foldername='16objects', beaching_strategy='1', flow='parabolic', d2cmax=0.03)
    # run(foldername='16objects', beaching_strategy='1', d2cmax=0.03)
    # run(beaching_strategy='1', flow='parabolic', d2cmax=0.03)
    # run(beaching_strategy='1', d2cmax=0.03)
    # run(foldername='16objects', beaching_strategy='3', flow='parabolic', d2cmax=0.03)
    # run(foldername='16objects', beaching_strategy='3', d2cmax=0.03)
    # run(beaching_strategy='3', flow='parabolic', d2cmax=0.03)
    # run(beaching_strategy='3', d2cmax=0.03)


