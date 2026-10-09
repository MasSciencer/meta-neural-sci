import meep as mp
import numpy as np

def refraction_input(input_angle_in_deg, epsilon):
    """
    Convert an incidence angle in air to the propagation angle inside the substrate
    using Snell's law.

    :param input_angle_in_deg: Incidence angle in degrees (int, float, or list).
    :param epsilon: Relative permittivity of the medium.
    :return: Angle in radians inside the medium, with the same type as the input.
    """
    n2 = np.sqrt(epsilon)

    # Convert to an array for consistent operations.
    input_array = np.atleast_1d(input_angle_in_deg)
    input_radians = np.radians(input_array)

    output_radians = np.arcsin(np.sin(input_radians) / n2)

    # Return the same type as the input.
    if isinstance(input_angle_in_deg, (list, np.ndarray)):
        return output_radians.tolist()
    else:
        return float(output_radians)

# setting oblique throughout the amplitude function amp_func
"""
(1) https://meep.readthedocs.io/en/master/Python_Tutorials/Mode_Decomposition/#reflectance-and-transmittance-spectra-for-planewave-at-oblique-incidence
(2) https://www.learnelectricalengineering.com/computational_em/lesson3_1.php
"""
k_point_normal = mp.Vector3(0, 0, 0) # normal incident

class FocusedEigenmodeSource:
    def __init__(self, diaphragm, frequency, center, df, focus_point, verbose=False):
        """
        Use an EigenModeSource to create a wavefront
        focused at distance f along the x axis.

        :param diaphragm: Aperture diameter along y.
        :param frequency: Source frequency.
        :param center: Source position (mp.Vector3).
        :param df: Source frequency bandwidth.
        :param focus_point: mp.Vector3
        :param verbose: Whether to print source parameters.
        """
        self.diaphragm = diaphragm
        self.frequency = frequency
        self.center = center
        self.df = df
        self.focus_distance = focus_point.x
        self.verbose = verbose

        # Wavenumber in the medium.
        self.k0 = 2 * np.pi * frequency

        self._configure_source()
        if self.verbose:
            print(self.describe())

    def _configure_source(self):
        def lens_phase(r):
            """
            Ideal lens phase profile without amplitude attenuation.
            The focal point is (x=f, y=0).
            """
            dx = self.focus_distance - r.x
            dy = r.y
            R = np.sqrt(dx ** 2 + dy ** 2)  # Distance to the focal point.
            return np.exp(-1j * self.k0 * R)

        self.sources = mp.EigenModeSource(
            src=mp.GaussianSource(self.frequency, self.df),
            center=self.center,
            size=mp.Vector3(0, self.diaphragm, 0),
            direction=mp.AUTOMATIC,
            eig_band=1,
            eig_parity=mp.EVEN_Y + mp.ODD_Z,
            eig_vol=None,
            amp_func=lens_phase
        )

    def get_sources(self):
        return self.sources

    def describe(self):
        return (f"FocusedEigenmodeSource(f={self.frequency}, "
                f"focus_distance={self.focus_distance}, "
                f"center={self.center}, "
                f"aperture={self.diaphragm})")

class BatchKvectorSource:
    """
    Accept an array of frequencies and the remaining source parameters,
    then create one KvectorSource per frequency.
    """
    def __init__(self, frequencies, dfreq, diaphragm, center, angle_in_rad):
        self.frequencies = np.asarray(frequencies)
        self.dfreq       = dfreq
        self.diaphragm   = diaphragm
        self.center      = center
        self.angle_in_rad       = angle_in_rad
        # Create the source objects.
        self._instances = [
            KvectorSource(
                frequency=f,
                dfreq=self.dfreq,
                diaphragm=self.diaphragm,
                center=self.center,
                angle_in_rad=self.angle_in_rad
            ) for f in self.frequencies
        ]

    def get_sources(self):
        """
        Return a flat list of all mp.Source objects.
        """
        # Flatten list-valued get_sources() results;
        # otherwise, append each source directly.
        result = []
        for inst in self._instances:
            src = inst.get_sources()
            # Handle list-valued get_sources() results.
            if isinstance(src, (list, tuple)):
                result.extend(src)
            else:
                result.append(src)
        return result

    def get_k_vectors(self):
        """
        Return the wave vector for each source.
        """
        return [inst.get_k_vector() for inst in self._instances]

    def __iter__(self):
        """
        Allow direct iteration over the list of mp.Source objects.
            for src in BatchKvectorSource(...): ...
        """
        return iter(self.get_sources())


class BatchEigenmodeSource:
    """
    Accept an array of frequencies and
    create one EigenmodeSource per frequency.
    """

    def __init__(self, frequencies, df, diaphragm, center, resolution, angle_in_rad, epsilon, verbose):
        self.frequencies = np.asarray(frequencies)
        self.df = df
        self.diaphragm = diaphragm
        self.resolution = resolution
        self.center = center
        self.angle_in_rad = angle_in_rad
        self.epsilon = epsilon
        self.verbose = verbose
        # Create the source objects.
        self._instances = [
            EigenmodeSource(
                diaphragm=self.diaphragm,
                angle_in_rad=self.angle_in_rad,
                frequency=f,
                df = self.df,
                resolution = self.resolution,
                center=self.center,
                epsilon = self.epsilon,
                verbose = self.verbose
            ) for f in self.frequencies
        ]

    def get_sources(self):
        """
        Return a flat list of all mp.Source objects.
        """
        # Flatten list-valued get_sources() results.
        result = []
        for inst in self._instances:
            src = inst.get_sources()
            # Handle list-valued get_sources() results.
            if isinstance(src, (list, tuple)):
                result.extend(src)
            else:
                result.append(src)
        return result

    def get_k_vectors(self):
        """
        Return the wave vector for each source.
        """
        return [inst.get_k_vector() for inst in self._instances]

    def __iter__(self):
        """
        Allow direct iteration over the list of mp.Source objects.
            for src in BatchKvectorSource(...): ...
        """
        return iter(self.get_sources())


class KvectorSource:
    def __init__(self, diaphragm, angle_in_rad, frequency, dfreq, center):
        self.diaphragm = diaphragm
        self.angle = angle_in_rad
        self.frequency = frequency
        self.dfreq = dfreq
        self.center = center
        self._configure_source()
        self.k_vector = mp.Vector3(x=np.cos(self.angle), y=np.sin(self.angle), z = 0).scale(self.frequency)

    def _configure_source(self):
        self.sources = mp.Source(mp.GaussianSource(frequency=self.frequency, fwidth=self.dfreq),
                             component=mp.Ez,
                             center=self.center,
                             size=mp.Vector3(x=0, y=self.diaphragm, z=0),
                             amp_func=self.tilt_function
                            )

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
    def __init__(self, diaphragm, angle_in_rad, frequency, resolution, center, df, epsilon, focus_point=None, verbose=False):
        """
        Unified source class for plane wave or ideally focused eigenmode sources.

        :param diaphragm: Diameter of the aperture (along y).
        :param angle_in_rad: Angle of incidence in radians (used only for plane wave mode).
        :param frequency: Central frequency.
        :param resolution: Simulation resolution (used for eig_vol in angled plane wave mode).
        :param center: Position of the source (mp.Vector3).
        :param df: Frequency width for Gaussian source.
        :param epsilon: Dielectric constant of the medium.
        :param focus_point: Optional mp.Vector3 for focusing point; if provided, switches to focused mode.
        :param verbose: If True, prints source description.
        """
        self.diaphragm = diaphragm
        self.angle = angle_in_rad
        self.frequency = frequency
        self.resolution = resolution
        self.center = center
        self.df = df
        self.epsilon = epsilon
        self.focus_point = focus_point
        self.verbose = verbose

        # Wave vector for plane wave mode
        self.k_vector = mp.Vector3(np.sqrt(self.epsilon) * self.frequency, 0, 0).rotate(
            mp.Vector3(0, 0, 1), self.angle)

        # for focus case
        self.k0 = 2 * np.pi * self.frequency * np.sqrt(self.epsilon)

        self._configure_source()
        if self.verbose:
            print(self.describe())

    def _configure_source(self):
        if self.focus_point is None:
            # Plane wave configuration
            if self.angle == 0:
                self.direction = mp.AUTOMATIC
                self.eig_parity = mp.EVEN_Y + mp.ODD_Z
                self.eig_vol = None
            else:
                self.direction = mp.NO_DIRECTION
                self.eig_parity = mp.ODD_Z
                self.eig_vol = mp.Volume(
                    center=self.center,
                    size=mp.Vector3(0, 1 / self.resolution, 0)
                )

            self.sources = mp.EigenModeSource(
                src=mp.GaussianSource(self.frequency, self.df),
                center=self.center,
                size=mp.Vector3(0, self.diaphragm, 0),
                direction=self.direction,
                eig_kpoint=self.k_vector,
                eig_band=1,
                eig_parity=self.eig_parity,
                eig_vol=self.eig_vol
            )
        else:
            # Focused configuration (assumes normal incidence)
            def lens_phase(r):
                """
                Phase profile for ideal lens focusing (unit magnitude, no amplitude attenuation).
                Focus at self.focus_point.
                """
                dx = self.focus_point.x - r.x
                dy = r.y
                R = np.sqrt(dx ** 2 + dy ** 2)
                return np.exp(-1j * self.k0 * R)

            self.sources = mp.EigenModeSource(
                src=mp.GaussianSource(self.frequency, self.df),  # Use Gaussian for consistency
                center=self.center,
                size=mp.Vector3(0, self.diaphragm, 0),
                direction=mp.AUTOMATIC,
                eig_band=1,
                eig_parity=mp.EVEN_Y + mp.ODD_Z,
                eig_vol=None,
                amp_func=lens_phase
            )

    def get_sources(self):
        return self.sources

    def get_k_vector(self):
        return self.k_vector

    def describe(self):
        return {
            'frequency': self.frequency,
            'center': self.center,
            'angle (rad)': self.angle,
            'k_vector': self.k_vector,
            'direction': self.direction,
            'eig_parity': self.eig_parity,
            'eig_vol': self.eig_vol
        }

    def __repr__(self):
        return f"EigenmodeSource(freq={self.frequency}, center={self.center}, angle={self.angle}, k={self.k_vector})"


# class EigenmodeSource:
#     def __init__(self, diaphragm, angle_in_rad, frequency, resolution, center, df, epsilon, verbose=False):
#         self.diaphragm = diaphragm
#         self.angle = angle_in_rad
#         self.frequency = frequency
#         self.resolution = resolution
#         self.center = center
#         self.k_vector = mp.Vector3(np.sqrt(epsilon) * self.frequency, 0, 0).rotate(
#             mp.Vector3(0, 0, 1), self.angle)
#         self.df = df
#         self.verbose = verbose
#         self._configure_source()
#         if self.verbose:
#             print(self.describe())
#
#     def _configure_source(self):
#         if self.angle == 0:
#             self.direction = mp.AUTOMATIC
#             self.eig_parity = mp.EVEN_Y + mp.ODD_Z
#             self.eig_vol = None
#         else:
#             self.direction = mp.NO_DIRECTION
#             self.eig_parity = mp.ODD_Z
#             self.eig_vol = mp.Volume(
#                 center=self.center,
#                 size=mp.Vector3(0, 1 / self.resolution, 0)
#             )
#
#         self.sources = mp.EigenModeSource(
#             src=mp.GaussianSource(self.frequency, self.df),
#             center=self.center,
#             size=mp.Vector3(0, self.diaphragm, 0),
#             direction=self.direction,
#             eig_kpoint=self.k_vector,
#             eig_band=1,
#             eig_parity=self.eig_parity,
#             eig_vol=self.eig_vol
#         )
#
#     def get_sources(self):
#         return self.sources
#
#
#     def get_k_vector(self):
#         return self.k_vector
#
#
#     def describe(self):
#         return {
#             'frequency': self.frequency,
#             'center': self.center,
#             'angle (rad)': self.angle,
#             'k_vector': self.k_vector,
#             'direction': self.direction,
#             'eig_parity': self.eig_parity,
#             'eig_vol': self.eig_vol
#         }
#
#
#     def __repr__(self):
#         return f"EigenmodeSource(freq={self.frequency}, center={self.center}, angle={self.angle}, k={self.k_vector})"