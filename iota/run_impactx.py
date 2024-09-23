"""
Tested with impact-x fork: https://github.com/cchall/impactx
"""

import numpy as np

import amrex.space3d as amr
from impactx import Config, ImpactX, elements, distribution
from rsbeams.rsstats import kinematic
################


sim = ImpactX()

# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

### Reference Particle
bunch_charge_C = 10.0e-15  # used if space charge
total_energy = 0.15e9

ref = sim.particle_container().ref_particle()
ref.set_charge_qe(-1.0).set_mass_MeV(0.510998950).set_kin_energy_MeV(149.48900105)
qm_eev = -1.0 / 0.510998950 / 1e6  # electron charge/mass in e / eV
ref.z = 0

### Distribution from MAD-X

particles = np.load('bunched_beam.bunch.npy')


x = np.array(particles[:, 0])
y = np.array(particles[:, 2])
z = np.array(particles[:, 4] * constants.c) 
px = np.array(particles[:, 1])
py = np.array(particles[:, 3])
reference = {'momentum': 149999129.59771833, 'velocity': 299790718.3997369, 'energy': 150000000.0, 'kenergy': 149489001.05, 'betagamma': 293.54097419910227, 'brho': 0.5003432394477326, 'beta': 0.9999941973181222, 'gamma': 293.54267753387757, 'p_unit': 'eV/c', 'e_unit': 'eV', 'mass': 510998.94999999995, 'mass_unit': 'eV/c^2', 'input': 150000000.0, 'input_unit': 'eV', 'input_type': 'energy'}
pz = -1*(particles[:, 5] - reference['gamma']) / reference['betagamma']

### Add distribution to simulation
pc = sim.particle_container()
if not Config.have_gpu:  # initialize using cpu-based PODVectors
    dx_podv = amr.PODVector_real_std()
    dy_podv = amr.PODVector_real_std()
    dt_podv = amr.PODVector_real_std()
    dpx_podv = amr.PODVector_real_std()
    dpy_podv = amr.PODVector_real_std()
    dpt_podv = amr.PODVector_real_std()
else:  # initialize on device using arena/gpu-based PODVectors
    dx_podv = amr.PODVector_real_arena()
    dy_podv = amr.PODVector_real_arena()
    dt_podv = amr.PODVector_real_arena()
    dpx_podv = amr.PODVector_real_arena()
    dpy_podv = amr.PODVector_real_arena()
    dpt_podv = amr.PODVector_real_arena()

for p_dx in x:
    dx_podv.push_back(p_dx)
for p_dy in y:
    dy_podv.push_back(p_dy)
for p_dt in z:
    dt_podv.push_back(p_dt)
for p_dpx in px:
    dpx_podv.push_back(p_dpx)
for p_dpy in py:
    dpy_podv.push_back(p_dpy)
for p_dpt in pz:
    dpt_podv.push_back(p_dpt)

pc.add_n_particles(
    dx_podv, dy_podv, dt_podv, dpx_podv, dpy_podv, dpt_podv, qm_eev, bunch_charge_C
)

### Lattice import
# Notebook lattice_setup.ipynb needs to be run to create this file
sim.lattice.load_file('in.madx', nslice=1)

def lattice_length(lattice):
    s = 0
    for ele in lattice:
        s+= ele.ds

    return s

for ele in sim.lattice:
    if ele.__class__.__name__ == 'ShortRF':
        print('Updating ShortRF')
        ele.V /= ref.mass_MeV
        if ele.freq < 0:
            # ele.freq contains harm * -1 in this case
            # calculate the time for a revolution * harm
            ele.freq = -1 * reference['velocity'] / (lattice_length(sim.lattice) / reference['velocity'] * ele.freq)


### Run 1 turn
sim.periods = 1
sim.evolve()

pc = sim.particle_container()
df = pc.to_df(local=True).to_hdf('diags/final_distribution.h5', 'final')

sim.finalize()
