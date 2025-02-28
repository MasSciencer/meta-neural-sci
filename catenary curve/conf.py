""""
Notice:
- by default, any place where there are no objects there is air (ε=1)
 although this can be changed by setting the default_material variable.
- all units in microns cause characteristic_length defined as 1 micron
"""
import meep as mp
import math

# CONSTANTS
characteristic_length = 1e-6 # 1 micron. Define the size of every single chunk (delta_x)
c = 3e8 # velocity of light
S = 0.707 # Courant factor for 2D


# 1. Settings for defining meta-atom parameters. Describe the shape of your meta-atom.

"""
    :parameter pml: PML for simulation cell. （Определяет толщину абсолютно поглощающего слоя, который ограничивает зону вычисления)
    :parameter in_sub: It usually depends on the wavelength and production technology.
    :parameter in_gap1: gap between left side pml and substrate
    :parameter in_gap2: gap between right side pml and atom
    :parameter in_epsilon: permittivity of substrate
    :parameter at_epsilon: permittivity of meta-atoms
    :parameter at_hei: meta-atom height (equals Y axis in global axis)
    :parameter unit_cell: the side of the square field for the unit cell
    
"""
"""
Note about materials:
    Сапфир (Al₂O₃):
        Реальная часть эпсилона: около 3.1-3.2.
        Мнимая часть эпсилона: очень малая, что делает сапфир почти без потерь в ИК-диапазоне.

    Селенид цинка (ZnSe):
        Реальная часть эпсилона: примерно 6.5-7.
        Мнимая часть эпсилона мала, что указывает на низкие оптические потери и хорошую прозрачность в ИК-диапазоне.
"""

# tune substrate
in_sub = 1.0  # micron
in_gap1 = 0.5 # micron
in_gap2 = 2.0 # micron
in_epsilon = 6.5
in_material = mp.Medium(epsilon=in_epsilon) # n≈sqrl(epsilon)

# tune meta-atom layer
unit_cell = 5  # micron
at_hei = 2  # micron
at_epsilon = 6.5
at_material = mp.Medium(epsilon=at_epsilon) # https://refractiveindex.info/?shelf=main&book=Si&page=Li-293K

# tune cell
pml = 3.0  # micron
cell_X = pml + in_gap1 + in_sub + at_hei + in_gap2 + pml # The X size of cell simulation.
cell_Y = unit_cell # The Y size of cell simulation.
sub_center = mp.Vector3(x=-0.5*cell_X+pml+in_gap1+0.5*in_sub)
at_center = mp.Vector3(x=-0.5*cell_X+pml+in_gap1+in_sub+at_hei*0.5)

# 2. Settings for determining the radiation source

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
    
"""

k_point = mp.Vector3(0, 0, 0) # normal incident
inc_wavelength = 3.7 # micron
inc_fcen = 1 / inc_wavelength  # Hz center frequency
inc_df = 0.05 * inc_fcen  # frequency width
real_wavelength = 3.7 * characteristic_length # 3.7e-6 m
real_frequency = 1 / real_wavelength # Hz
source_center = mp.Vector3(x=-0.5*cell_X+pml) # micron
flux_center = mp.Vector3(x=-0.5*cell_X+pml+in_gap1+in_sub+at_hei+0.5*in_gap2)

# 3. Optimisation setting
"""
    :parameter chunk_time: time to across 1 chunk by light source
    :parameter real_time_move: time to across simulation cell from source plane to end of cell
    :parameter min_resolution: the smallest resolution of simulation with getting rid of missing proper shape of
     sin's wave curve. However I recommend increase this quantity up to several times in final simulation. 
    :parameter end_time: minimum of simulation time of MEEP's units to rich an edge of cell
          
"""
chunk_time = characteristic_length/(c*S) # second
real_time_move = (cell_X - source_center.x)*1e-6/c # second
min_resolution = math.ceil(8 * characteristic_length / real_wavelength / math.sqrt(at_epsilon)) # pixels/μm
end_time = math.ceil(real_time_move/chunk_time) # MEEP steps

# 4. PSF

"""
    :parameter resolution: number of pixels per image. Higher quality of image than bigger resolution
    
"""

resolution_img = 1000 # higher quality of image than bigger resolution. Number of pixels per image


# 5. Check outputs
print(f"substrait begin in {sub_center.x-0.5*in_sub} and end {sub_center.x + 0.5*in_sub}")
print(f"atom begin in {at_center.x-0.5*at_hei} and end {at_center.x + 0.5*at_hei}")
print(f"flux_center begin in {flux_center.x}")
print(f"at_center is {at_center.x}")
print(f"sub_center is {sub_center.x}")
print(f"resolution is {min_resolution}")
print(f"end_time is {end_time}")