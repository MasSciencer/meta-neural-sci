""""
Notice:
- by default, any place where there are no objects there is air (ε=1)
 although this can be changed by setting the default_material variable.
- all units in microns cause characteristic_length defined as 1 micron
"""
import meep as mp
import math
import numpy as np
from lib import help_functions

# GENERAL
is_diaphragm_auto =       True  # adjusts the aperture size to ensure the required angle discreteness
is_tan_law =              False # the law of ray displacement
is_increase_cell =        False # only necessary to visualize the near fields in the form of animation
"""
Examples of basic settings sets

# f = 2.0 * 1e3 # focal length in                                         [micron]
# aperture = 20.0 * 1e3 # size in                                         [micron]
# inc_start_wavelength = 4.0 #                                            [micron]
# inc_end_wavelength = 4.5 #                                              [micron]
# n_angles = 6 #                                                          [integer]
# gap_from_pinhole_to_surface = 2.0*1e3

# f = 0.5 * 1e3 # focal length in                                        [micron]
# aperture = 1.5 * 1e3 # size аin                                        [micron]
# gap_from_pinhole_to_surface = 0.3 * 1e3
# inc_start_wavelength = 3.7 #                                           [micron]
# inc_end_wavelength = 4.8 #                                             [micron]
# n_angles = 2 #                                                         [integer]
# n_frequencies = 1

# f = 0.15 * 1e3 # focal length in                                       [micron]
# aperture = 0.15 * 1e3 # size in                                        [micron]
# gap_from_pinhole_to_surface = 0.0 * 1e3
# inc_start_wavelength = 4.0 #                                           [micron]
# inc_end_wavelength = 4.5 #                                             [micron]
# n_angles = 1 #                                                         [integer]
# n_frequencies = 1

"""
f = 2.0 * 1e3 # focal length in                                        [micron]
aperture = 2.0 * 1e3 # size аin                                        [micron]
gap_from_pinhole_to_surface = 0.0 * 1e3
inc_start_wavelength = 3.7 #                                           [micron]
inc_end_wavelength = 4.8 #                                             [micron]
n_angles = 1
n_frequencies = 5

diaphragm = int(help_functions.find_D_pinhole(n_angles, aperture)) if is_diaphragm_auto else 0.5 * 1e3
add_size = f if is_increase_cell else 0

if n_angles == 1:
    max_angle = 0,
    angle_list = [0]
else:
    max_angle =  help_functions.get_max_angle_in_deg(aperture, diaphragm, gap_from_pinhole_to_surface),
    angle_list = help_functions.get_angle_list(n_angles, diaphragm, gap_from_pinhole_to_surface)

inc_wavelength = (inc_start_wavelength + inc_end_wavelength) / 2
numerical_aperture = np.round((diaphragm / math.sqrt(diaphragm**2 + 4 * f**2)), decimals=2)

coef_resolution = 10 # minimum is min_resolution, usually equals 1. Default is 10.
sim_resolution = coef_resolution
ff_res = 5 # multiplier of the number of points in the far field. Default = 10
spot_ff = 200
order_step = inc_wavelength * 2
pure_orders = int((aperture / 2) // order_step) # initial orders. if is_split = True be recalculated
loc_orders = int((diaphragm) // order_step)

inc_dwavelength = 0.05 * inc_wavelength  # [micron]
orders = pure_orders + 1 # the number of radial layers + 1
stack = 2 # Defines the number of sub-cells. At 0, it has only one main cell.
Q = 80 # Curve detailing. It has almost no effect on the result
w = 0.4 # 400 nm. The width of the gap between the curves
LIMIT_SHAPE = 0.2 # 200 nm. sections less than 200 nm are removed.
order_gap = 1.0 # Defines the minimum distance between orders.
maximum_run_time = 200

# OPTIMIZATION
dx = 1/coef_resolution
x_dim = int((inc_wavelength*2) / dx)
h_dim = 30              # 30 pixels
patience = 3            # number of steps without significant improvement
threshold = 0.00001     # the minimum gain of f0 considered significant

# CONSTANTS
characteristic_length = 1e-6 # 1 micron. Define the size of every single chunk (delta_x)
c = 3e8 # velocity of light
S = 0.707 # Courant factor for 2D

# 1. Settings of meta-atom parameters. Describe the shape of your meta-atom.

"""
    :parameter pml: PML for simulation cell.
    :parameter in_sub: It usually depends on the wavelength and production technology.
    :parameter in_gap1: gap between left side pml and substrate
    :parameter in_gap2: gap between right side pml and atom
    :parameter in_epsilon: permittivity of substrate
    :parameter at_epsilon: permittivity of meta-atoms
    :parameter at_hei: meta-atom height (equals Y axis in global axis)
    :parameter unit_cell: the side of the square field for the unit cell
"""

# tune substrate
in_gap1 = 4 # micron
in_gap2 = 4 # micron
in_epsilon = 3.54**2
in_material = mp.Medium(epsilon=in_epsilon) # n≈sqrl(epsilon)
# in_material = Al2O3_aniso

# tune meta-atom layer
at_hei = 3.0  # Initial height [micron]
at_epsilon = 3.54**2
at_material = mp.Medium(epsilon=at_epsilon) # https://refractiveindex.info/?shelf=main&book=Si&page=Franta

# tune cell
pml = 4.0  # [micron]
cell_X = pml + in_gap1 + at_hei + in_gap2 + pml # The X size of cell simulation.
cell_Y = aperture # The Y size of cell simulation.
in_sub = cell_X / 2  # micron
sub_center = mp.Vector3(x=-in_sub/2 -add_size/2)
at_center = mp.Vector3(x = 0-add_size/2)
cell = mp.Vector3(cell_X+add_size, cell_Y, 0)  # размер области симуляции
loc_cell = mp.Vector3(x = cell.x, y = diaphragm, z = 0)

# 2. Settings of sources

"""
    :parameter k_point: a wave vector is a vector used in describing a wave, with a typical unit 
    being cycle per metre. It has a magnitude and direction. Its magnitude is the wavenumber of the wave 
    (inversely proportional to the wavelength), and its direction is perpendicular to the wavefront. 
    In isotropic media, this is also the direction of wave propagation.
    :parameter inc_wavelength: the normalized wavelength value in the MEEP
    :parameter inc_fcen: the normalized frequency value in the MEEP
    :parameter inc_df: frequency range relative to the center
    :parameter source_center: location of the wave source
    :parameter flux_center: monitor location
    :parameter min_frequency: This means we need to make sure that kx is always smaller than vector |k| https://www.learnelectricalengineering.com/computational_em/lesson2_8.php
    no matter what angle we choose kx, will always be smaller than vector |k| for all frequencies in our simulation - we're safe. 
"""
inc_nfreq = n_frequencies # frequency discreteness
inc_df = 1/inc_dwavelength
inc_fcen = 1/inc_wavelength  # Hz center frequency

inc_start_frequency = 1/inc_start_wavelength
inc_end_frequency = 1/inc_end_wavelength

if inc_nfreq == 1:
    inc_wavelength_array = np.array([inc_wavelength])
    inc_frequents_array = np.array([inc_fcen])
else:
    inc_wavelength_array = np.linspace(inc_start_wavelength, inc_end_wavelength, inc_nfreq)
    inc_wavelength_array = np.round(inc_wavelength_array, 2)
    # inc_wavelength_array = inc_wavelength_array[1:]
    inc_frequents_array = np.linspace(inc_start_frequency, inc_end_frequency, inc_nfreq)
    # inc_frequents_array = inc_frequents_array[1:]

min_frequency = inc_fcen - inc_df/2
real_wavelength = inc_wavelength * characteristic_length # 4.25e-6 m
real_frequency = 1 / real_wavelength # Hz
main_source_center = mp.Vector3(x=-cell_X/2 + pml -add_size/2 , y = 0)
flux_center = mp.Vector3(x=-0.5*cell_X+pml+in_gap1+at_hei+0.9*in_gap2 -add_size/2)

# EFFICIENCY
airy_diameter = 1.22 * inc_wavelength_array / numerical_aperture

# 3. Optimisation setting
"""
    :parameter chunk_time: time to across 1 chunk by light source
    :parameter real_time_move: time to across simulation cell from source plane to end of cell
    :parameter min_resolution: the smallest resolution of simulation with getting rid of missing proper shape of
     sin's wave curve. However I recommend increase this quantity up to several times in final simulation. 
    :parameter till_edge_time_in_free_space: minimum of simulation time of MEEP's units to rich an edge of cell
    :parameter till_power_target: threshold for triggering ending
    
    end_functions:
    :parameter run_end_energy: function for ending simulation by energy
    :parameter run_end_time: function for ending simulation by time. Break a simulation when wave tach a monitor + 1 step        
"""
chunk_time = characteristic_length/(c*S) # second
real_time_move = (cell_X - main_source_center.x)*1e-6/c # second
min_resolution = math.ceil(8 * characteristic_length / real_wavelength / math.sqrt(at_epsilon)) # pixels/μm
till_edge_time_in_free_space = math.ceil(real_time_move/chunk_time) # MEEP steps for n=1, if n not equal 1 that time should be increased
till_power_target = 1e-3 # power units

run_end_energy = lambda sim: sim.run(until_after_sources=mp.stop_when_fields_decayed(
    till_edge_time_in_free_space,
    mp.Ez,
    mp.Vector3(x=flux_center.x, y=cell_Y, z=0),
    till_power_target)
)
run_end_time = lambda sim: sim.run(until=math.ceil(5 + real_time_move / chunk_time / math.cos(np.radians(angle_list[0])))) # use: conf.run_end_time(sim)

# 4. PSF
resolution_base = 201 # Higher quality of image than bigger resolution. Number of pixels per image


