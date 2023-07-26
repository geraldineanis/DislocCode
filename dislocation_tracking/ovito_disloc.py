import numpy as np
import os
import time
from natsort import natsorted

from ovito.io import import_file, export_file
from ovito.modifiers import DislocationAnalysisModifier
from ovito.data import DislocationNetwork

import ovito
ovito.enable_logging()

start_frame = 25000

# Set dump file writing frequency
step = 500

# dumps = natsorted(os.listdir("./dumps"))

# Load structure(s)
pipeline = import_file("./dumps/dump.shear_unwrap.*")
data = pipeline.compute()

# Set up dislocation analysis modifier
modifier = DislocationAnalysisModifier()
modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.FCC
pipeline.modifiers.append(modifier)

# Evaluate pipeline at every frame
# Write dislocation output files for analysis
start = time.time()
for frame in range(pipeline.source.num_frames):
    data = pipeline.compute(frame)

    # Extract outputs
    # Number of dislocations found
    ndisloc = len(data.dislocations.segments)

    # Number and coordinates of vertices
    lens = []
    vertices = []
    for segment in data.dislocations.segments:
        lens.append(len(segment.points))
        vertices.append(segment.points)

    with open(f"./Ni_disloc/disloc_data_{start_frame+(frame*step)}.txt", "w") as fdata:
        # Write the number of dislocations found
        print(f"Writing disloc_data_{start_frame+(frame*step)}.txt")
        fdata.write(f"{ndisloc} \n")

        for j, length in enumerate(lens):
            # Dislocation index
            fdata.write(f"{j} \n")
            # Number of vertices
            fdata.write(f"{length} \n")
            # Coordinates of vertices
            for coords in vertices[j]:
                fdata.write(f"{coords[0]} {coords[1]} {coords[2]} \n")
        fdata.close()

        print(f"Written disloc_data_{frame*step}.txt")

    # Export dislocation lines to a CA file:
    export_file(data, f"./Ni_disloc/dislocations_{frame*step}.ca", "ca")
end = time.time()
print(f"Execution time = {round(end-start,2)}")

#print(len(data.dislocations.segments[0].points))


# total_line_length = data.attributes['DislocationAnalysis.total_line_length']
# cell_volume = data.attributes['DislocationAnalysis.cell_volume']

#print("Dislocation density: %f" % (total_line_length / cell_volume))

# # Print list of dislocation lines:
# print("Found %i dislocation segments" % len(data.dislocations.segments))
# for segment in data.dislocations.segments:
#     print("Segment %i: length=%f, Burgers vector=%s" % (segment.id, segment.length, segment.true_burgers_vector))
#     print(segment.points)

# # Export dislocation lines to a CA file:
# export_file(pipeline, "dislocations.ca", "ca")

## Or export dislocations to a ParaView VTK file:
#export_file(pipeline, "dislocations.vtk", "vtk/disloc")
