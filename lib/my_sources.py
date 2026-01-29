import meep as mp
import numpy as np

def refraction_input(input_angle_in_deg, epsilon):
    """
    Переводит угол падения из воздуха в угол распространения внутри подложки
    по закону преломления.

    :param input_angle_in_deg: угол падения в градусах (int, float или list)
    :param epsilon: диэлектрическая проницаемость среды
    :return: угол в радианах внутри среды (того же типа, что и входной угол)
    """
    n2 = np.sqrt(epsilon)

    # Преобразуем в массив для унифицированных операций
    input_array = np.atleast_1d(input_angle_in_deg)
    input_radians = np.radians(input_array)

    output_radians = np.arcsin(np.sin(input_radians) / n2)

    # Возвращаем результат того же типа, что и вход
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
        Источник с использованием EigenModeSource, формирующий фронт,
        фокусирующийся в точку на расстоянии f по оси z.

        :param diaphragm: диаметр апертуры (вдоль y)
        :param frequency: частота
        :param center: положение источника (mp.Vector3)
        :param df: ширина источника
        :param focus_point: mp.Vector3
        :param verbose: вывод параметров
        """
        self.diaphragm = diaphragm
        self.frequency = frequency
        self.center = center
        self.df = df
        self.focus_distance = focus_point.x
        self.verbose = verbose

        # волновое число в среде
        self.k0 = 2 * np.pi * frequency

        self._configure_source()
        if self.verbose:
            print(self.describe())

    def _configure_source(self):
        def lens_phase(r):
            """
            Фазовый профиль идеальной линзы (без амплитудного затухания).
            Фокус в точке (x=f, y=0).
            """
            dx = self.focus_distance - r.x
            dy = r.y
            R = np.sqrt(dx ** 2 + dy ** 2)  # путь до фокуса
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
    Принимает массив частот и все остальные параметры,
    под капотом создаёт по KvectorSource на каждую частоту.
    """
    def __init__(self, frequencies, dfreq, diaphragm, center, angle_in_rad):
        self.frequencies = np.asarray(frequencies)
        self.dfreq       = dfreq
        self.diaphragm   = diaphragm
        self.center      = center
        self.angle_in_rad       = angle_in_rad
        # создаём объекты-источники
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
        Возвращает плоский список всех mp.Source.
        """
        # если KvectorSource.get_sources() возвращает список,
        # можно сделать flatten, иначе просто collect:
        result = []
        for inst in self._instances:
            src = inst.get_sources()
            # если get_sources() — список:
            if isinstance(src, (list, tuple)):
                result.extend(src)
            else:
                result.append(src)
        return result

    def get_k_vectors(self):
        """
        Для каждого источника возвращает его волновой вектор.
        """
        return [inst.get_k_vector() for inst in self._instances]

    def __iter__(self):
        """
        Чтобы можно было итерироваться прямо по списку mp.Source:
            for src in BatchKvectorSource(...): ...
        """
        return iter(self.get_sources())


class BatchEigenmodeSource:
    """
    Принимает массив частот
    создаёт по EigenmodeSource на каждую частоту.
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
        # создаём объекты-источники
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
        Возвращает плоский список всех mp.Source.
        """
        # если KvectorSource.get_sources() возвращает список
        result = []
        for inst in self._instances:
            src = inst.get_sources()
            # если get_sources() — список:
            if isinstance(src, (list, tuple)):
                result.extend(src)
            else:
                result.append(src)
        return result

    def get_k_vectors(self):
        """
        Для каждого источника возвращает его волновой вектор.
        """
        return [inst.get_k_vector() for inst in self._instances]

    def __iter__(self):
        """
        Чтобы можно было итерироваться прямо по списку mp.Source:
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