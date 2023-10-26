import numpy as np
import os
import time
from natsort import natsorted

from ovito.io import import_file, export_file
from ovito.modifiers import DislocationAnalysisModifier
from ovito.data import DislocationNetwork

import ovito
ovito.enable_logging()

start_frame = 0

# Set dump file writing frequency - every n timesteps
step = 500

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
