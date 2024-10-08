import numpy as np

import amrex.space3d as amr
from impactx import Config, ImpactX, elements, distribution
from rsbeams.rsstats import kinematic
from rsbeams.rsdata.SDDS import readSDDS
from scipy import constants
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
ref.set_charge_qe(-1.0).set_mass_MeV(0.510998950).set_kin_energy_MeV(kinematic.Converter(energy=total_energy)(silent=True)['kenergy']*1e-6)
qm_eev = -1.0 / 0.510998950 / 1e6  # electron charge/mass in e / eV
ref.z = 0

### Distribution from MAD-X

with open('../ptc_particles.madx', 'r') as ff:
    ptc_particles = ff.readlines()

quants = ('x', 'px', 'y', 'py', 't', 'pt')

elegant_distribution_file = readSDDS('../elegant_matched/bunched_beam.bunch.sdds')
elegant_distribution_file.read()
elegant_distribution = elegant_distribution_file.columns.squeeze()

x = np.array(elegant_distribution['x'])
y = np.array(elegant_distribution['y'])
z = np.array(elegant_distribution['t'] * constants.c) 
px = np.array(elegant_distribution['xp'])
py = np.array(elegant_distribution['yp'])
reference = kinematic.Converter(energy=total_energy)()
pz = -(kinematic.Converter(
    betagamma=elegant_distribution['p']
)(silent=True)['gamma'] - reference['gamma']) / reference['betagamma']

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
sim.lattice.load_file('../madx/in.madx', nslice=1, beamline='IOTA')


### Run 1 turn
sim.periods = 1
sim.evolve()

pc = sim.particle_container()
df = pc.to_df(local=True).to_hdf('diags/final_distribution.h5', 'final')

sim.finalize()