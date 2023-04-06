import numpy as np

diameter = 1.0

# Define the distance range to calculate the potential energy
r_min = 0.0
r_max = 2.0 * diameter
dr = 0.01
r_vals = np.arange(r_min, r_max+dr, dr)

# Calculate the potential energy as a function of distance
energy_vals = np.zeros_like(r_vals)
for i in range(len(r_vals)):
    if r_vals[i] < diameter:
        energy_vals[i] = np.inf
        

# Write the potential energy values to a file
with open('hard_sphere.table', 'w') as f:
    f.write('# distance (sigma) potential energy (kcal/mol)\n')
    for i in range(len(r_vals)):
        f.write('{:.6f} {:.6f} 0.0 0.0\n'.format(r_vals[i], energy_vals[i]))
