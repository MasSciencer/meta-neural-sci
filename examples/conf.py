""""
Notice:
- by default, any place where there are no objects there is air (ε=1)
 although this can be changed by setting the default_material variable.
- all units in microns cause characteristic_length defined as 1 micron
"""
import meep as mp
import math
import numpy as np
from meep.materials import Al2O3_aniso

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
in_gap1 = 1 # micron
in_gap2 = 3.0 # micron
in_epsilon = 6.5
in_refractive_indec = math.sqrt(in_epsilon)
in_material = mp.Medium(epsilon=in_epsilon) # n≈sqrl(epsilon)
# in_material = Al2O3_aniso

# tune meta-atom layer
unit_cell = 5  # micron
at_hei = 2  # micron
at_epsilon = 6.5
at_refractive_indec = math.sqrt(at_epsilon)
at_material = mp.Medium(epsilon=at_epsilon) # https://refractiveindex.info/?shelf=main&book=Si&page=Li-293K
# at_material = Al2O3_aniso

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
    :parameter min_frequency: This means we need to make sure that kx is always smaller than vector |k| https://www.learnelectricalengineering.com/computational_em/lesson2_8.php
    no matter what angle we choose kx, will always be smaller than vector |k| for all frequencies in our simulation - we're safe. 
"""

inc_wavelength = 3.7 # micron
inc_fcen = 1 / inc_wavelength  # Hz center frequency
inc_df = 0.2 * inc_fcen  # frequency width
inc_nfreq = 1 # frequency discreteness
inc_angle = 0 # degrees
min_frequency = inc_fcen - inc_df/2
real_wavelength = inc_wavelength * characteristic_length # 4.25e-6 m
real_frequency = 1 / real_wavelength # Hz
source_center = mp.Vector3(x=-0.5*cell_X+pml) # micron
flux_center = mp.Vector3(x=-0.5*cell_X+pml+in_gap1+in_sub+at_hei+0.5*in_gap2)


class DefSource:
    def __init__(self, y):
        self.y = y
        self._configure_source()
    def _configure_source(self):
        self.sources = [mp.Source(
            mp.GaussianSource(frequency=inc_fcen, fwidth=inc_df),
            component=mp.Ez,
            center=source_center,
            size=mp.Vector3(x=0, y=self.y, z=0),
            )]
    def get_sources(self):
        return self.sources
    def get_k_vector(self):
        k_vector_normal = mp.Vector3(0, 0, 0)  # normal incident
        return k_vector_normal

# 2.1 setting oblique throughout the amplitude function amp_func
"""
(1) https://meep.readthedocs.io/en/master/Python_Tutorials/Mode_Decomposition/#reflectance-and-transmittance-spectra-for-planewave-at-oblique-incidence
(2) https://www.learnelectricalengineering.com/computational_em/lesson3_1.php
"""
k_point_normal = mp.Vector3(0, 0, 0) # normal incident
class KvectorSource:
    def __init__(self, y, angle=0):
        self.y = y
        self.angle = np.radians(angle)
        self._configure_source()
        self.k_vector = mp.Vector3(x=np.cos(self.angle), y=np.sin(self.angle), z = 0).scale(inc_fcen)

    def _configure_source(self):
        self.sources = [mp.Source(mp.ContinuousSource(frequency=inc_fcen, is_integrated=True),
                             component=mp.Ez,
                             center=source_center,
                             size=mp.Vector3(x=0, y=self.y, z=0),
                             amp_func=self.tilt_function
                             )]

    def tilt_function(self, vec):
        return np.exp(1j * 2 * np.pi * (self.k_vector.x * vec.x + self.k_vector.y * vec.y + self.k_vector.z * vec.z))
    def get_sources(self):
        return self.sources

    def get_k_vector(self):
        return self.k_vector

# 2.2  setting oblique throughout eigenmode source oblique
"""
(1) https://meep.readthedocs.io/en/master/Python_Tutorials/Eigenmode_Source/#planewaves-in-homogeneous-media
"""
class EigenmodeSource:
    def __init__(self, y, resolution=8, angle=0):
        self.y = y
        self.resolution = resolution
        self.angle = np.radians(angle)
        self.k_vector = mp.Vector3(at_refractive_indec * inc_fcen, 0, 0).rotate(
            mp.Vector3(0, 0, 1),
            self.angle
        )
        self._configure_source()

    def _configure_source(self):
        if self.angle == 0:
            self.direction = mp.AUTOMATIC
            self.eig_parity = mp.EVEN_Y + mp.ODD_Z
            self.symmetries = [mp.Mirror(mp.Y)]
            self.eig_vol = None
        else:
            self.direction = mp.NO_DIRECTION
            self.eig_parity = mp.ODD_Z
            self.symmetries = []
            self.eig_vol = mp.Volume(
                center=source_center,
                size=mp.Vector3(0, 1 / self.resolution, 0)
            )

        self.sources = [
            mp.EigenModeSource(
                src=mp.ContinuousSource(inc_fcen),
                center=source_center,
                size=mp.Vector3(0, self.y, 0),
                direction=self.direction,
                eig_kpoint=self.k_vector,
                eig_band=1,
                eig_parity=self.eig_parity,
                eig_vol=self.eig_vol
            )
        ]

    def get_sources(self):
        return self.sources

    def get_k_vector(self):
        return self.k_vector

# 3. Optimisation setting
"""
    :parameter chunk_time: time to across 1 chunk by light source
    :parameter real_time_move: time to across simulation cell from source plane to end of cell
    :parameter min_resolution: the smallest resolution of simulation with getting rid of missing proper shape of
     sin's wave curve. However I recommend increase this quantity up to several times in final simulation. 
    :parameter till_edge_time_in_free_space: minimum of simulation time of MEEP's units to rich an edge of cell
    :parameter till_power_target: threshold for triggering ending
    :parameter end_energy: function for ending simulation by energy
    :parameter end_time: function for ending simulation by time
          
"""
chunk_time = characteristic_length/(c*S) # second
real_time_move = (cell_X - source_center.x)*1e-6/c # second
min_resolution = math.ceil(8 * characteristic_length / real_wavelength / math.sqrt(at_epsilon)) # pixels/μm
till_edge_time_in_free_space = math.ceil(real_time_move/chunk_time) # MEEP steps for n=1, if n not equal 1 that time should be increased
till_power_target = 1e-3 # power units


""" end_functions"""
end_energy = lambda sim: sim.run(until_after_sources=mp.stop_when_fields_decayed(
    till_edge_time_in_free_space,
    mp.Ez,
    mp.Vector3(x=flux_center.x, y=cell_Y, z=0),
    till_power_target)
)

end_time = lambda sim: sim.run(until=math.ceil(real_time_move / chunk_time) / math.cos(inc_angle))


# 4. PSF

"""
    :parameter resolution: number of pixels per image. Higher quality of image than bigger resolution
    
"""

resolution_img = 800 # Higher quality of image than bigger resolution. Number of pixels per image


# 5. Check outputs
print(f"substrait begin in {sub_center.x-0.5*in_sub} and end {sub_center.x + 0.5*in_sub}")
print(f"atom begin in {at_center.x-0.5*at_hei} and end {at_center.x + 0.5*at_hei}")
print(f"flux_center begin in {flux_center.x}")
print(f"at_center is {at_center.x}")
print(f"sub_center is {sub_center.x}")
print(f"resolution is {min_resolution}")
print(f"end_time is {math.ceil(real_time_move / chunk_time) / math.cos(inc_angle)}")