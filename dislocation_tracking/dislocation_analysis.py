import numpy as np
from matplotlib import pyplot as plt
import math
import sys
from collections import defaultdict

def read_disloc_data(disloc_file):
    """
    Read in dislocation position data obtained from 
    processing MD trajectory using OVITO sctript provided.

    Parameters
    ----------
    disloc_file : str
                  Dislocation data file name.
    
    Returns
    -------
    numpy.ndarray
                  Dislocation line x, y, and z coordinates.
    """
    with open(disloc_file) as f:
        ndisloc = int(f.readline())     # number of dislocations    
        indices = []                    # indices of dislocations 
        lens = []                       # lengths of dislocations
        dislocations = []               # coordinates of dislocations

        # Read dislocation coordinates for each dislocation found
        for i in range(ndisloc):
            disloc_i = []
            index = indices.append(int(f.readline().strip()))
            length = int(f.readline().strip())
            lens.append(length)
            for i in range(length):
                coords = f.readline().strip()
                coords = np.array(coords.split())
                coords = coords.astype(np.float64)
                disloc_i.append(coords)
            dislocations.append(disloc_i)

    return np.array(dislocations, dtype=object)

def get_cell_lims(dump_file):
    """
    Extract simulation cell extents from a LAMMPS dump file.

    Parameters
    ----------
    dump_file : str
                LAMMPS dump file name.

    Returns
    -------
    float
                Simulation cell length in x (Angstroms).
    float
                Simulation cell length in y (Angstroms).
    float
                Simulation cell length in z (Angstroms).            
    """
    dump = np.loadtxt(f"{dump_file}", skiprows=5, max_rows=3)
    lims = []
    for i in range(3):
        lims.append(dump[i][1] - dump[i][0])
    return lims[0], lims[1], lims[2]

def wrap_coords(dislocation, ax, extent):
    """
    Function to wrap the coordinates of a dislocation line along a given axis.

    Parameters
    ----------
    dislocation : numpy.ndarray
                  Dislocation line coordinates.
    ax          : str
                  "x", "y", or "z" - direction for wrapping.
    extent      : float
                  Cell length (in Angstroms) in direction of wrapping.

    Returns
    -------
    numpy.ndarray
                  Modified dislocation coordinates after wrapping.
    """
    if ax == "x":
        j = 0
    elif ax == "y":
        j = 1
    elif ax == "z":
        j = 2
    else:
        print("Error: ax must be set to 'x', 'y', or 'z'")
        sys.exit()

    for i in range(len(dislocation)):
        if dislocation[i][j] < 0.0:
            dislocation[i][j] = dislocation[i][j] + extent
        if dislocation[i][j] > extent:
            dislocation[i][j] = dislocation[i][j] - extent
    
    return dislocation

def get_disloc_coords(dislocation):
    """
    Extract and split coordinates of a dislocation line.

    Parameters
    ----------
    dislocation : numpy.ndarray
                  Dislocation line coordinates.
    
    Returns
    -------
    numpy.ndarray
                  Dislocation line x coordinates.
    numpy.ndarray
                  Dislocation line y coordinates.
    numpy.ndarray
                  Dislocation line z coordinates.                                    
    """
    x_coords = np.array(dislocation, dtype=object)[:,0]
    y_coords = np.array(dislocation, dtype=object)[:,1]
    z_coords = np.array(dislocation, dtype=object)[:,2]

    return x_coords, y_coords, z_coords

def get_avg_pos(dislocation):
    """
    Calculate average position of a dislocation line.
    
    Parameters
    ----------
    dislocation : numpy.ndarray
                  Dislocation line coordinates.
    
    Returns
    -------
    numpy.ndarray
                  Average dislocation line coordinates in x, y, and x.
    """
    x_coords, y_coords, z_coords = get_disloc_coords(dislocation)
    avg_pos = [np.average(x_coords),np.average(y_coords),np.average(z_coords)]

    return np.array(avg_pos)

def perfect_disloc_coords(partial_1, partial_2):
    """
    Calculate the x-coordinates of a perfect dislocation 
    as the average of the x-coordinates of two partial dislocations
    at every simulation timestep.

    Parameters
    ----------
    partial_1 : list
                First partial dislocation x-coordinates timeseries.
    partial_2 : list
                Second partial dislocation x-coordinates timeseries.                
                
    Returns
    -------
    numpy.ndarray
                Perfect dislocation x-coordinates timeseries.                
    """
    return np.array([np.average([partial_1[i], partial_2[i]]) for i in range(len(partial_1))])

def get_velocity(disloc_x_pos, timestep):
    """
    Function to calculate a dislocation line's x-component of velocity in Angstroms/ps.

    Parameters
    ----------
    disloc_x_pos : list
                   Dislocation line x-coordinates timeseries.
    timestep     : float
                   Simulation timestep in ps.

    """
    return np.gradient(disloc_x_pos, timestep)

def write_prop(filename, property, time):
    """
    Function to write out a property time series
    of a dislocation to a text file.

    Parameters
    ----------
    filename : str
               Output file name.
    property : numpy.ndarray
               Property timeseries.
    time     : numpy.ndarray
               Timesteps.
    """
    f = open(f"{filename}", "w")
    for i in range(len(property)):
        f.write(f"{round(time[i],1)}     {property[i]} \n")
    f.close()

def track_disloc(avg, x_lim):

    """
    Track dislocation lines across multiple frames.
    - Dislocations are tracked by checking the minimum distance moved by a dislocation between frames. 
    
    - Adapted from the approach in:
    https://pysource.com/2021/10/05/object-tracking-from-scratch-opencv-and-python/
    
    - Assumptions:
    1. The dislocation only moves in the x direction
    2. The number of dislocations is constant across all frames 
    i.e. no dislocations are generated nor lost throughout the analysed trajectory.
    
    Parameters
    ----------
    avg      : numpy.ndarray
               Average x, y, and z coordinates of dislocation lines in the analysed trajectory.
    x_lim    : float
               Simulation cell length in x (Angstroms).
    n_frames : int
               Number of frames in trajectory.
    
    Returns
    -------
    list
               Sorted dislocation x-coordinates across trajectory frames.
    list
               Sorted coordinates of dislocation line vertices across frames.
    list
               Timesteps.
    
    """
    # Tracking ID
    track_ID = 0

    # Dictionary to store dislocations
    tracking_objects = defaultdict(list)

    # Compare x coordinate of average position
    # first we only compare the first and second frames
    # to determine object id at beginning
    # This sorts the first two points well

    # Cutoff distance
    d_cutoff = 5.0 # Angstroms

    for i in range(1,2):
        for j in range(len(avg[i])):
            for k in range(len(avg[i])):
                displacement = avg[i][j][0][0] - avg[i-1][k][0][0]
                if abs(displacement) < d_cutoff:
                    pbc = False
                    tracking_objects[track_ID].append([i-1,np.array([avg[i-1][k][0][0], avg[i-1][k][0][1], avg[i-1][k][0][2]]),int(pbc),avg[i-1][k][1]]) # first frame coordinates
                    tracking_objects[track_ID].append([i,np.array([avg[i][j][0][0], avg[i][j][0][1], avg[i][j][0][2]]),int(pbc),avg[i][j][1]])   # second frame coordinates
                    track_ID += 1          
                else:
                    displacement = avg[i][j][0][0] + x_lim - avg[i-1][k][0][0]
                    if abs(displacement) < d_cutoff:
                        pbc = True
                        tracking_objects[track_ID].append([i-1,np.array([avg[i-1][k][0][0], avg[i-1][k][0][1], avg[i-1][k][0][2]]),int(pbc),avg[i-1][k][1]]) # first frame coordinates
                        tracking_objects[track_ID].append([i,np.array([avg[i][j][0][0], avg[i][j][0][1], avg[i][j][0][2]]),int(pbc),avg[i][j][1]])   # second frame coordinates
                        track_ID += 1
                     
    # For the remaining frames we compare the current x coordinate
    # to those of the already existing objects
    for i in range(2,len(avg)):
        for j in range(len(avg[i])):
            d_i = []
            for obj_ID, obj in tracking_objects.items():
                # Calculate the distances between the current avg x position
                # and the last x position added to each tracking object
                dist_1 = avg[i][j][0][0] - obj[-1][1][0]
                dist_2 = avg[i][j][0][0] + x_lim - obj[-1][1][0]
                dist_3 = avg[i][j][0][0] - x_lim - obj[-1][1][0]

                # Determine whether dislocation has crossed periodic boundary            
                if abs(dist_2) < abs(dist_1) and abs(dist_2) < abs(dist_3):
                    dist = abs(dist_2)
                    pbc = 1
                elif abs(dist_3) < abs(dist_1) and abs(dist_3) < abs(dist_2):
                    dist = abs(dist_3)
                    pbc = 3
                else:
                    dist = abs(dist_1)
                    pbc = 0

                d_i.append([avg[i][j][0][0], avg[i][j][0][1], avg[i][j][0][2], obj_ID, dist, pbc])

            # Append coordinate to corresponding dislocation
            # Find the minimum distance between the current average position
            # and the average position in each dislocation in the previous frame
            d_i = np.array(d_i)
            min_dist = np.amin(d_i, axis=0)[4]
            indx = np.where(d_i == min_dist)[0][0]
            x = d_i[indx][0]
            y = d_i[indx][1]
            z = d_i[indx][2]
            pbc = int(d_i[indx][5])

            tracking_objects[indx].append([i, np.array([x,y,z]), pbc, avg[i][j][1]])

    # Correct for PBCs
    pbcs = []
    for i, object in tracking_objects.items():
        pbcs_i = []
        for j in range(len(object)):
            pbcs_i.append(object[j][2])
        pbcs.append(np.array(pbcs_i))
  
    starts = []
    test = []
    for pbc in pbcs:
        starts.append(np.where(pbc == 1)[0])
        test.append(np.where(pbc ==3)[0])


    for i, start in enumerate(starts):
        for s in start:
            for j in range(s,len(pbcs[i])):
                tracking_objects[i][j][1][0] += x_lim

    for i, start in enumerate(test):
        for s in start:
            for j in range(s,len(pbcs[i])):
                tracking_objects[i][j][1][0] -= x_lim
             
    # Collect position data after tracking
    position = []
    vertices = []
    t = []
    for i, object in tracking_objects.items():
        position_i = []
        vertices_i = []
        t_i = []
        for j in range(len(object)):
            position_i.append(math.hypot(object[j][1][0]))
            vertices_i.append(object[j][3])
            t_i.append(object[j][0]*0.001*500)
        position.append(position_i)
        vertices.append(vertices_i)
        t.append(t_i)

    # Change vertices from tuples to arrays for easier handling
    # Sort vertices along y coords
    disloc = []
    for i in range(len(vertices)):
        frames = []
        for j in range(len(vertices[i])):
            vertex = []
            for k in range(len(vertices[i][j])):
                # Wrap y
                v_x = vertices[i][j][k][0]
                v_y = vertices[i][j][k][1]
                v_z = vertices[i][j][k][2]                           
                vertex.append([v_x, v_y, v_z])
            frames.append(sorted(vertex, key=lambda v: v[1]))
        disloc.append(frames)

    # Unwrap x to be consistent with average position data
    for i, start in enumerate(starts):

        for s in start:
            for j in range(s,len(pbcs[i])):
                for k in range(len(disloc[i][j])):
                    disloc[i][j][k][0] += x_lim

    return position, disloc, t