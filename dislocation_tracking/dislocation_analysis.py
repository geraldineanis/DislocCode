from xml.dom.expatbuilder import FragmentBuilderNS
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
        # index
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

def get_velocity(avg_position, del_t):
    """
    Function to calculate the velocity
    using central differences
    given the average dislocation position
    as input
    """
    vel = []

    # LHS boundary node
    vel.append((avg_position[1]-avg_position[-1])/(2.0*del_t))
    # RHS boundary node
    vel.append((avg_position[0]-avg_position[-2])/(2.0*del_t))
    # Bulk
    for i in range(1,len(avg_position)-1):
        vel.append((avg_position[i+1] - avg_position[i-1])/(2.0*del_t))

    return vel

def moving_average(a, n):
    """
    Function to calculate moving average
    """
    ma = np.cumsum(a, dtype=float)
    ma[n:] = ma[n:] - ma[:-n]
    ma = ma[n-1:]/n
    return ma

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
    coordinates
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
                displacement = avg[i][j][0] - avg[i-1][k][0]
                if abs(displacement) < d_cutoff:
                    pbc = False
                    tracking_objects[track_ID].append([i-1,avg[i-1][k],int(pbc)]) # first frame coordinates
                    tracking_objects[track_ID].append([i,avg[i][j],int(pbc)])   # second frame coordinates
                    track_ID += 1          
                else:
                    displacement = avg[i][j][0] + x_lim - avg[i-1][k][0]
                    if abs(displacement) < d_cutoff:
                        pbc = True
                        # Unwrap coordinates
                        # avg[i][j][0] += x_lim                           
                        tracking_objects[track_ID].append([i-1,avg[i-1][k],int(pbc)]) # first frame coordinates
                        tracking_objects[track_ID].append([i,avg[i][j],(pbc)])   # second frame coordinates
                        track_ID += 1   

    # For the remaining frames we compare the current x coordinate
    # to those of the already existing objects
    distances = []
    for i in range(2,len(avg)):
        print(i)
        distances_i = []
        for j in range(len(avg[i])):
            # if avg[i][j][0] > x_lim:
            #     avg[i][j][0] -= x_lim            
            # if avg[i][j][0] < 0.0:
            #     avg[i][j][0] += x_lim                   
            print(avg[i][j])
            d_i = []
            for obj_ID, obj in tracking_objects.items():
                print("obj[-1][1][0] ", obj[-1][1][0])
                del_t = 0.001*500 # ps
                # Calculate the distances between the current avg x position
                # and the last x position added to each tracking object
                # we need to account for PBCs                    
                dist_1 = avg[i][j][0] - obj[-1][1][0]
                dist_2 = avg[i][j][0] + x_lim - obj[-1][1][0]
                print("dist_1 ",dist_1)
                print("dist_2 ",dist_2)

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
                d_i.append([avg[i][j][0], avg[i][j][1], avg[i][j][2],obj_ID, dist, pbc])
            print(np.array(d_i))
            distances_i.append(d_i)
            # append coordinate to corresponding dislocation
            d_i = np.array(d_i)
            min_dist = np.amin(d_i, axis=0)[4]
            print(min_dist)
            indx = np.where(d_i == min_dist)[0][0]
            x = d_i[indx][0]
            y = d_i[indx][1]
            z = d_i[indx][2]
            pbc = int(d_i[indx][5])
            print(indx)
            print(x, y, z)
            print(pbc)
            tracking_objects[indx].append([i,np.array([x,y,z]),pbc])
            print("\n")

        distances.append(distances_i)

    print(tracking_objects[0][0][1][0])

    # Correct for PBCs
    pbcs = []
    for i, object in tracking_objects.items():
        pbcs_i = []
        for j in range(len(object)):
            pbcs_i.append(object[j][2])
        pbcs.append(np.array(pbcs_i))
    
    print(pbcs[1])
    print(np.where(pbcs[1] == 1))
    
    starts = []
    for pbc in pbcs:
        starts.append(np.where(pbc == 1)[0])

    print(starts)
    print(len(starts))

    for i, start in enumerate(starts):
        for s in start:
            for j in range(s,len(pbcs[i])):
                tracking_objects[i][j][1][0] += x_lim
             
    # Collect position data after tracking
    position = []
    t = []
    for i, object in tracking_objects.items():
        position_i = []
        t_i = []
        for j in range(len(object)):
            position_i.append(math.hypot(object[j][1][0]))
            t_i.append(object[j][0]*0.001*500)
        position.append(position_i)
        t.append(t_i)

    # print("Tracking success rate")
    # for i in range(len(position)):
    #     success = (len(position[i])/n_frames)*100.0
    #     print(f"Partial {i+1}: {success}%")
    # print("\n")

    # # # plt.hist(acc)

    return position, t


###################################################################
#                                                                 #
#                              Plotting                           #
#                                                                 #
###################################################################
