import numpy as np
import math
from collections import defaultdict
from matplotlib import pyplot as plt
import seaborn as sns

###################################################################
#                                                                 #
#                               General                           #
#                                                                 #
###################################################################

def read_disloc_data(disloc_file):
    """
    Function to read in dislocation data
    and convert into a numpy array for
    further processing
    """
    with open(disloc_file) as f:
        ndisloc = int(f.readline())
        indices = []
        lens = []
        dislocations = []
        # Read in first dislocation data
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
    Function to extract cell extents from dump file
    """
    dump = np.loadtxt(f"{dump_file}", skiprows=5, max_rows=3)
    lims = []
    for i in range(3):
        lims.append(dump[i][1] - dump[i][0])
    return lims[0], lims[1], lims[2]

def wrap_coords_x(dislocation, x_extent):
    """
    Function to wrap the coordinates of a dislocation line
    given the dislocation the coordinates of the dislocation line vertices
    and the cell extents in x
    """
    for i in range(len(dislocation)):
        if dislocation[i][0] < 0.0:
            dislocation[i][0] = dislocation[i][0] + x_extent
        if dislocation[i][0] > x_extent:
            dislocation[i][0] = dislocation[i][1] - x_extent

    return dislocation

def wrap_coords_y(dislocation, y_extent):
    """
    Function to wrap the coordinates of a dislocation line
    given the dislocation the coordinates of the dislocation line vertices
    and the cell extents in y
    """
    for i in range(len(dislocation)):
        if dislocation[i][1] < 0.0:
            dislocation[i][1] = dislocation[i][1] + y_extent
        if dislocation[i][1] > y_extent:
            dislocation[i][1] = dislocation[i][1] - y_extent

    return dislocation

def wrap_coords_z(dislocation, z_extent):
    """
    Function to wrap the coordinates of a dislocation line
    given the dislocation the coordinates of the dislocation line vertices
    and the cell extents in z
    """
    for i in range(len(dislocation)):
        if dislocation[i][2] < 0.0:
            dislocation[i][2] = dislocation[i][2] + z_extent
        if dislocation[i][2] > z_extent:
            dislocation[i][2] = dislocation[i][2] - z_extent

    return dislocation

def get_disloc_coords(dislocation):
    """
    Function to get the coordinates of a dislocation line
    """
    x_coords = np.array(dislocation, dtype=object)[:,0]
    y_coords = np.array(dislocation, dtype=object)[:,1]
    z_coords = np.array(dislocation, dtype=object)[:,2]

    return x_coords, y_coords, z_coords

def get_avg_pos(dislocation):
    """
    Function to average positions of a dislocation
    line given the vertices as input
    Input must be given as a numpy array
    """
    x_coords, y_coords, z_coords = get_disloc_coords(dislocation)
    avg_pos = [np.average(x_coords),np.average(y_coords),np.average(z_coords)]

    return avg_pos

def perfect_disloc_coords(partial_1, partial_2):
    """
    Function to calculate the coordinates of a perfect dislocation
    by averaging the coordinates of two partial dislocations
    """
    return [np.average([partial_1[i], partial_2[i]]) for i in range(len(partial_1))]

def get_velocity(disloc, timestep):
    """
    Function to calculate velocity from position data
    Works both for partial or perfect dislocation
    """
    return np.gradient(disloc, timestep)

def write_prop(filename, property, time):
    """
    Function to write out property time series of a dislocation to a text file

    """
    f = open(f"{filename}", "w")
    for i in range(len(property)):
        f.write(f"{round(time[i],1)}     {property[i]} \n")
    f.close()

###################################################################
#                                                                 #
#                        Dislocation Tracking                     #
#                                                                 #
###################################################################

def track_disloc(avg, x_lim, n_frames):

    """
    Function to track dislocation lines across multiple using average 
    positions as input.
    - For tracking the dislocations, we use a simple object tracking
    algorithm that checks the change in x coordinates between the
    current and previous frame to track a dislocation and assign it
    a unique ID
    - 
    - The algorithm is adapted from the approach in:
    https://pysource.com/2021/10/05/object-tracking-from-scratch-opencv-and-python/
    - Assumptions:
    1. The dislocation only moves in the x direction
    and that it is therefore sufficient to only use the change in the x
    coordinates..+
    2. The number of dislocations is constant across all frames
    i.e. no dislocations are generated nor lost throughout the
    trajectory analysed (this will be changed in the future)
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
            # if avg[i][j][0] > x_lim:
            #     avg[i][j][0] -= x_lim            
            # if avg[i][j][0] < 0.0:
            #     avg[i][j][0] += x_lim                   
            # print(avg[i][j])
            d_i = []
            for obj_ID, obj in tracking_objects.items():
                # Calculate the distances between the current avg x position
                # and the last x position added to each tracking object
                dist_1 = avg[i][j][0][0] - obj[-1][1][0]
                dist_2 = avg[i][j][0][0] + x_lim - obj[-1][1][0]

                # Determine whether dislocation has crossed periodic boundary            
                # For the first i frames the dislocation is allowed to move back
                # This accounts for the first few frames before the dislocation
                # starts moving                        
                if i <= 10:
                    if dist_2 < abs(dist_1):
                        dist = dist_2
                        pbc = True
                    else:
                        dist = abs(dist_1)
                        pbc = False
                else:
                    if dist_2 < abs(dist_1) or dist_1 < 0.0:
                        dist = dist_2
                        pbc = True
                    else:
                        dist = abs(dist_1)
                        pbc = False
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
    for pbc in pbcs:
        starts.append(np.where(pbc == 1)[0])

    for i, start in enumerate(starts):
        for s in start:
            for j in range(s,len(pbcs[i])):
                tracking_objects[i][j][1][0] += x_lim
             
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


###################################################################
#                                                                 #
#                              Plotting                           #
#                                                                 #
###################################################################
