import meep as mp
import numpy as np
import conf

class DefSource:
    def __init__(self, y):
        self.y = y
        self._configure_source()
    def _configure_source(self):
        self.sources = [mp.Source(
            mp.GaussianSource(frequency=conf.inc_fcen, fwidth=conf.inc_df),
            component=mp.Ez,
            center=conf.source_center,
            size=mp.Vector3(x=0, y=self.y, z=0),
            )]
    def get_sources(self):
        return self.sources
    def get_k_vector(self):
        k_vector_normal = mp.Vector3(0, 0, 0)  # normal incident
        return k_vector_normal

# setting oblique throughout the amplitude function amp_func
"""
(1) https://meep.readthedocs.io/en/master/Python_Tutorials/Mode_Decomposition/#reflectance-and-transmittance-spectra-for-planewave-at-oblique-incidence
(2) https://www.learnelectricalengineering.com/computational_em/lesson3_1.php
"""
k_point_normal = mp.Vector3(0, 0, 0) # normal incident
class KvectorSource:
    def __init__(self, frequency, y, angle=0):
        self.y = y
        self.angle = np.radians(angle)
        self.frequency = frequency
        self._configure_source()
        self.k_vector = mp.Vector3(x=np.cos(self.angle), y=np.sin(self.angle), z = 0).scale(self.frequency)

    def _configure_source(self):
        self.sources = [mp.Source(mp.ContinuousSource(frequency=self.frequency, end_time= conf.add_time/2, is_integrated=True),
                             component=mp.Ez,
                             center=conf.source_center,
                             size=mp.Vector3(x=0, y=self.y, z=0),
                             amp_func=self.tilt_function),
                        mp.Source(mp.ContinuousSource(frequency=self.frequency, end_time= conf.add_time/2, is_integrated=True),
                            component=mp.Ey,
                            center=conf.source_center,
                            size=mp.Vector3(x=0, y=self.y, z=0),
                            amp_func=lambda vec: 1j * self.tilt_function(vec)  # Phase shift Pi/4
                            )]

    def tilt_function(self, vec):
        return np.exp(1j * 2 * np.pi * (self.k_vector.x * vec.x + self.k_vector.y * vec.y + self.k_vector.z * vec.z))
    def get_sources(self):
        return self.sources

    def get_k_vector(self):
        return self.k_vector

# setting oblique throughout eigenmode source oblique
"""
(1) https://meep.readthedocs.io/en/master/Python_Tutorials/Eigenmode_Source/#planewaves-in-homogeneous-media
"""
class EigenmodeSource:
    def __init__(self, y, resolution=8, angle=0):
        self.y = y
        self.resolution = resolution
        self.angle = np.radians(angle)
        self.k_vector = mp.Vector3(conf.at_refractive_indec * conf.inc_fcen, 0, 0).rotate(
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
                center=conf.source_center,
                size=mp.Vector3(0, 1 / self.resolution, 0)
            )

        self.sources = [
            mp.EigenModeSource(
                src=mp.ContinuousSource(conf.inc_fcen),
                center=conf.source_center,
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