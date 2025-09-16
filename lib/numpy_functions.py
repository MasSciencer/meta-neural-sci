import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import os
from scipy.interpolate import interp1d
from scipy.fft import fft
from scipy.signal import savgol_filter
from typing import Union, Sequence,List, Optional, Iterable
from dataclasses import dataclass
from enum import Enum
import gdspy
import conf
from lib import my_sources
import trimesh
from shapely.geometry import Polygon
from shapely.ops import orient

class ResolutionMode(str, Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"

@dataclass(frozen=True)
class Resolution:
    """
    w: wide
    h: high
    """
    w: int
    h: int

class AdaptiveResolution:
    def __init__(
        self,
        drt: Iterable[float],
        src: ResolutionMode = ResolutionMode.MID,
        x_dim: Optional[int] = None,
        h_dim: int = 1
    ):
        """
        Definition: Если передается значение x_dim, то разрешение постоянное по всем ячейкам.
        :param drt: ширины сегментов по x
        :param src: (опционально) режим разрешения по x
        :param x_dim: (фиксированное) разрешение по x
        :param h_dim: разрешение по y
        """
        self.drt = list(drt)
        self.src = src
        self.x_dim = x_dim
        self.height_dim = h_dim

    def _base(self) -> int:
        mapping = {
            ResolutionMode.LOW: 200,
            ResolutionMode.MID: 500,
            ResolutionMode.HIGH: 1000,
        }
        try:
            return mapping[self.src]
        except KeyError:
            raise ValueError(f"src must be one of {list(mapping)}")

    def generate(self) -> List[Resolution]:
        # override по x_dim
        if self.x_dim is not None:
            return [Resolution(self.x_dim, self.height_dim) for _ in self.drt]

        base = self._base()
        first = self.drt[0]
        k = first / base

        # первый сегмент — просто (base, height_dim)
        result = [Resolution(base, self.height_dim)]
        # остальные — масштаб
        for w in self.drt[1:]:
            res_x = int(w // k)
            result.append(Resolution(res_x, self.height_dim))
        return result

    # для удобства: сделать объект итерируемым и индексируемым
    def __iter__(self):
        yield from self.generate()

    def __getitem__(self, idx):
        return self.generate()[idx]

def catenary_function(x, x0, a):
    """
    Description:
    We use the simplest CC description function: centered between two radii.

    :param x: x coordinate
    :param x0: center between radii
    :param a: coefficient of curvature
    """
    # https://chatgpt.com/share/67e139b7-8b94-800f-beef-0178c329f0d2

    return a * np.cosh((x - x0) / a)


def cut_y(xx, yy, y0):
    x_values = []
    for i in range(len(xx) - 1):
        y1, y2 = yy[i], yy[i + 1]
        if (y1 - y0) * (y2 - y0) <= 0:
            x1, x2 = xx[i], xx[i + 1]
            x_intersection = x1 + (y0 - y1) * (x2 - x1) / (y2 - y1)
            x_values.append(x_intersection)

    # Сортируем значения по возрастанию
    x_values.sort()

    return np.array(x_values)

def calculate_radii(D, f, wavelength, pure_orders, split):
    """
    Вычисляет радиусы на поверхности линзы, на которых происходит полный фазовый сдвиг от 0 до 2π,
    с возможностью дробления первого промежутка.

    :param D: Диаметр линзы.
    :param f: Фокусное расстояние линзы.
    :param wavelength: Длина волны света.
    :param pure_orders: Максимальное значение n для расчета радиусов.
    :param split: На сколько частей дробить первый радиус (r[0]).
                  Если <=1, дробление не производится.
    :return: radii: Список радиусов.
    """
    # Собираем базовый список радиусов для n = 1, ..., pure_orders
    radii = []
    for n in range(1, pure_orders + 1):
        r = np.sqrt((n * wavelength)**2 + 2 * n * wavelength * f)
        if r <= D / 2:
            radii.append(r)
        else:
            print(f"Предупреждение: {pure_orders + 1 - n} радиусов были отброшены. Превышают D/2.")
            break

    # Если требуется дробление первого радиуса
    if split and split > 1 and len(radii) > 0:
        first_r = radii[0]
        # Генерируем split точек: first_r * (1/split), (2/split), ..., (split/split == 1)*first_r
        subdivisions = [first_r * i / split for i in range(1, split + 1)]
        # Заменяем оригинальный первый радиус на все полученные части
        radii = subdivisions + radii[1:]

    return radii

def calculate_radii_step(D, orders):
    """
    Вычисляет набор радиусов, получающихся при равномерном делении радиуса линзы на orders частей.

    Args:
        D: Диаметр линзы
        orders: Количество сегментов радиуса (int или скалярный tensor)

    Returns:
        Тензор радиусов (float64), упорядоченный по возрастанию,
        размерностью [orders], в диапазоне (0, D/2].
    """
    # orders нам нужен как целое число для arange
    orders_int = int(orders)

    # Полный радиус
    r_max = D / 2

    # Индексы от 1 до orders_int
    n = np.arange(1, orders_int + 1)

    # Разбиваем радиус на равные части: r_i = (i / orders_int) * r_max
    radii = n * (r_max / orders_int)

    return radii

def get_drt_midrt(rt, orders, is_global = True):
    """
    :param rt: радиусы порядков
    :param orders: количество порядков
    :return: (list) ширины между радиусами и глобальные центры между радиусами
    """
    drt = []
    midrt = []

    if is_global:
        print(f"len(rt) is {len(rt)}")
        print(f"len(orders) is {orders}")
        for order in range(orders):
            drt.append(2 * rt[order] if order == 0 else (rt[order] - rt[order - 1]))
            midrt.append(0 if order == 0 else (rt[order] + rt[order - 1]) / 2)

    else:
        for order in range(0, orders+1):
            if order == 0:
                drt.append(rt[order])
                midrt.append(rt[order]/2 - rt[orders]/2)
            else:
                drt.append(rt[order] - rt[order - 1])
                midrt.append((rt[order] + rt[order - 1]) / 2 - rt[orders]/2)

        for order in range(0, orders):
            if order == 0:
                drt.append(rt[order])
                midrt.append(rt[order]/2)

            else:
                drt.append(rt[order] - rt[order - 1])
                midrt.append((rt[order] + rt[order - 1]) / 2)

    return drt, midrt

def create_2d_curve(a_params, orders, stack, rt, Q=200, w=0.1, plot=False):
    """
    :param orders: количество порядков в линзе
    :param stack: количество кривых, которые пересекаются сечением на каждом порядке. Четное число
    :param rt: лист с радиусами
    :param a_params: вектор или лист с коэффициентами формы
    :param Q: количество точек на кривую. Default 200 points
    :param w: The width of the gap between the curves. Default 100 nm
    :return: coordinates of X crossections points (as 2D numpy array) and centers
    """
    list_pure_x = []
    y0 = -0.0001 + 2 * stack * w

    if plot:
        setup_visualization()
        ax.axhline(y=y0, color='black', linestyle='--', linewidth=1)

    for m in range(orders):
        if m == 0:
            rt0, x0 = 0, 0
        else:
            rt0 = rt[m - 1] + 0.01
            x0 = (rt[m] + rt0) / 2  # центры кривых

        a = a_params[m]
        x = np.linspace(rt0, rt[m] - 0.01, Q)
        y = catenary_function(x, x0, a)  # Предполагается, что `a` определено

        # Идём в обратном порядке по стеку (stack, stack-1, ..., 0)
        for n in range(stack, -1, -1):
            ip = np.argmin(y)
            y_shifted = y - y[ip]
            ymax = 0.5 * (np.max(x) - np.min(x))

            y_top = np.clip(y_shifted, -ymax, ymax)
            y_above_zero = np.clip(y_shifted - w, -ymax, ymax)

            xx = np.concatenate([x, x[::-1]])
            yy = np.concatenate([y_top, y_above_zero[::-1]]) + 2 * n * w

            x_pure = cut_y(xx, yy, y0)  # Предполагается, что `cut_y` определена
            list_pure_x.append(x_pure)

            if plot:
                # Визуализация
                plot_curve(xx, yy, x_pure, y0, n)

    # Преобразование в 2D numpy массив с размерностью (orders, stack + 1)
    x_pure = np.array(list_pure_x, dtype=object).reshape(orders, stack + 1)

    return x_pure

def calculate_width_and_centers(x_pure):
    """
    Вычисляет разницы в зависимости от количества элементов во вложенных массивах.

    Параметры:
    x_pure -- 2D numpy массив, где каждый элемент — массив с 1, 2 или 4 числами

    Возвращает:
    numpy массив с результатами (форма сохраняется, содержимое заменяется на разницы)
    """
    # Создаем выходной массив той же формы, но с dtype=object для гибкости
    result_width = np.empty_like(x_pure, dtype=object)
    result_centers = np.empty_like(x_pure, dtype=object)

    for i in range(x_pure.shape[0]):
        for j in range(x_pure.shape[1]):

            arr = x_pure[i][j]

            if len(arr) == 1 and i == 0 and j == 0:
                result_width[i][j] = np.array([2 * arr[
                    0]])  # Пояснение, далее мы используем ширины которые центрированы. А 0 порядок это по сути урезанная ширина, так как другая сторона не учитывается. И чтобы ячейка не оказалась уменьшенной в 2 раза. Ее надо искусственно расширить.
                result_centers[i][j] = np.array([0])

            elif len(arr) == 2:
                width1 = arr[1] - arr[0]
                result_width[i][j] = np.array([width1])
                center1 = arr[0] + width1 / 2
                result_centers[i][j] = np.array([center1])

            elif len(arr) == 4:
                width1 = arr[1] - arr[0]
                width2 = arr[3] - arr[2]
                result_width[i][j] = np.array([width1, width2])
                center1 = arr[0] + width1 / 2
                center2 = arr[2] + width2 / 2
                result_centers[i][j] = np.array([center1, center2])

            else:
                result_width[i][j] = np.array([0])
                result_centers[i][j] = np.array([0])

    return result_centers, result_width


def get_width_and_center(shape_params, plot=True):
    """
    Description: возвращает numpy координаты центров и ширин для передачи в геометрию MEEP
    """
    rt = calculate_radii(conf.aperture, conf.f, conf.inc_wavelength, conf.pure_orders, conf.split)
    x_pure = create_2d_curve(shape_params, conf.orders, conf.stack, rt, conf.Q, conf.w, plot=plot)
    centers, width = calculate_width_and_centers(x_pure)
    # if plot:
    #     # Сбор данных для графиков
    #     stack_indices = list(range(width.shape[1]))
    #     orders_indices = list(range(width.shape[0]))
    #     width_values = [width[0][j][0] for j in stack_indices]
    #     center_values = [centers[j][0][0] for j in orders_indices]
    #
    #     # Создаем фигуру с двумя subplots
    #     plt.figure(figsize=(12, 6))
    #
    #     # График для ширины (левый subplot)
    #     plt.subplot(1, 2, 1)
    #     plt.plot(stack_indices, width_values, 'bo-', label=f'Width for {width.shape[1] - 1} stacks')
    #     plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    #     plt.xlabel('Stack Index')
    #     plt.ylabel('Width Value')
    #     plt.title('Width vs Stack Index')
    #     plt.legend()
    #     plt.grid()
    #
    #     # График для центров (правый subplot)
    #     plt.subplot(1, 2, 2)
    #     plt.plot(orders_indices, center_values, 'ro-', label=f'Centers for {width.shape[0]} orders')
    #     plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    #     plt.xlabel('Order Index')
    #     plt.ylabel('Center Value')
    #     plt.title('Center vs Order Index')
    #     plt.legend()
    #     plt.grid()
    #
    #     # Оптимизация расположения
    #     plt.tight_layout()
    #     plt.show()

    return centers, width

def get_segments(x_pure, rt):
    result_segments = np.empty_like(x_pure, dtype=object)

    for i in range(x_pure.shape[0]):
        mid_rt = (rt[i] + rt[i - 1]) / 2
        for j in range(x_pure.shape[1]):

            arr = x_pure[i][j]

            if len(arr) == 1 and i == 0 and j == 0:
                result_segments[i][j] = np.array([-arr[0], arr[0]])

            if len(arr) == 2 and i == 0:
                result_segments[i][j] = np.array([-arr[1], -arr[0], arr[0], arr[1]])

            elif len(arr) == 2:
                result_segments[i][j] = np.array([arr[0], arr[1]]) - mid_rt

            elif len(arr) == 4:
                result_segments[i][j] = np.array([arr[0], arr[1], arr[2], arr[3]]) - mid_rt
            else:
                ValueError('Out of test')

    return result_segments


def find_intersections(
        segment: np.ndarray,
        d_rt: float,
        nx_resolution: int,
        smooth: bool = True,  # Флаг для включения сглаживания
        beta: float = 10.0,  # Параметр "резкости" перехода
        plot: bool = False,
) -> np.ndarray:
    """
    Возвращает бинарную или сглаженную маску пересечений отрезков с вектором global_vec
    Нужно передать весь массив сегментов

    Параметры:
        segment (np.ndarray): массив с данными отрезков
                               (shape: [n_stacks, variable-length arrays])
        d_rt (float): Параметр для построения global_vec
        nx_resolution (int): Количество точек в векторе сканирования
        smooth (bool): Если True, возвращает плавные веса вместо 0/1.
        beta (float): Параметр резкости перехода (больше → резче).

    Return:
        Матрица размером (Nx, 1) Со значением 1 где есть материал, и 0 где его нет
    """
    global_vec = np.linspace(-0.5 * d_rt, 0.5 * d_rt, nx_resolution)
    result = np.zeros(nx_resolution, dtype=np.float32)  # Теперь float

    segments_stacks = segment[:]

    for stack_idx, points in enumerate(segments_stacks):
        pts = np.asarray(points).ravel()

        if pts.size == 2:
            starts_ends = [(pts[0], pts[1])]
        elif pts.size == 4:
            starts_ends = [(pts[0], pts[1]), (pts[2], pts[3])]
        else:
            continue

        for start, end in starts_ends:
            # Рассчитываем расстояние до отрезка [start, end]
            distances = []
            for x in global_vec:
                if x < start:
                    d = start - x  # Расстояние до левого конца
                elif x > end:
                    d = x - end  # Расстояние до правого конца
                else:
                    d = -1.0  # Внутри отрезка (отрицательное)
                distances.append(d)

            distances = np.array(distances)

            if smooth:
                # Применяем сигмоиду к расстояниям
                arg = -beta * (-distances)  # -distances: внутри → положительные
                arg = np.clip(arg, -500, 500)  # предотвращаем переполнение
                weights = 1 / (1 + np.exp(arg))
                result = np.maximum(result, weights)  # Объединяем пересечения
            else:
                # Бинарная маска (старая логика)
                mask = (global_vec >= start) & (global_vec <= end)
                result[mask] = 1.0

    if plot:
        # Визуализация
        plt.figure(figsize=(12, 4))
        plt.stem(np.linspace(-d_rt / 2, d_rt / 2, nx_resolution), result)
        plt.title("Карта пересечений")
        plt.xlabel("Позиция")
        plt.ylabel("Наличие пересечения")
        plt.show()

    return result

"""
SIMULATION
"""
def infill_geometry(shape_params, geometry):
    centers, width = get_width_and_center(shape_params, plot = True)
    # add sub
    geometry.append(mp.Block(
        material=conf.in_material,
        size=mp.Vector3(conf.in_sub, mp.inf, 0),
        center=mp.Vector3(conf.sub_center.x),
    ))
    # add atoms
    for i in range(conf.orders):
        for j in range(conf.stack + 1):
            for k in range (centers[i, j].size):
                if width[i][j][k] < conf.LIMIT_SHAPE: continue
                else:
                    ###---positive infilling---###
                    geometry.append(mp.Block(
                        material=conf.at_material,
                        size=mp.Vector3(x=conf.at_hei, y=width[i][j][k], z=0),
                        center=mp.Vector3(x=conf.at_center.x, y=centers[i][j][k], z=0)
                    ))
    return geometry


def setup_simulation(geometry, sources, df, nfreq, plot=True, coef_resolution=conf.coef_resolution):
    sim = mp.Simulation(
        cell_size=conf.cell,
        boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.X, side=mp.ALL)],
        geometry=geometry,
        sources=sources,
        resolution=conf.min_resolution * coef_resolution,
        symmetries=[mp.Mirror(mp.Y)]
    )

    # Adding a near2far area
    near2far_region = sim.add_near2far(
        conf.inc_fcen,  # центральная частота
        df,  # ширина спектра
        nfreq,  # дискретность частот
        mp.Near2FarRegion(center=conf.flux_center, size=mp.Vector3(y=conf.spot_ff)),
    )

    if plot:
        plt.figure(figsize=(1, 100))
        sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(0, conf.cell_Y / 4), size=mp.Vector3(conf.cell[0], conf.cell[1] / 2))
                   )
        plt.show()

    return sim, near2far_region

"""
INTENSITY
"""
def shift_y(angle, f = conf.f, is_tan_law = conf.is_tan_law):
    if is_tan_law:
        shift_y = f * np.tan(np.radians(angle))
    else:
        shift_y = f * np.sin(np.radians(angle))

    return shift_y

def get_abscissa (spot_ff, y, angle, shift = None):
    """
    :param spot_ff: область построения
    :param angle:
    :param y: ордината
    :param shift: опциональное добавочное значение сдвига
    """
    size_of_y = len(y)
    if shift != None: add_y = shift
    else: add_y = shift_y(angle)
    x = np.linspace(-spot_ff/2 + add_y, spot_ff/2 + add_y, size_of_y)
    return x


def compute_I_base(sim, dtf_obj, inc_nfreq):
    """
    Вычисляет значения I_base для каждой из inc_nfreq частот.

    Параметры:
    sim : объект симуляции MEEP
    dtf_obj : объект DFT
    inc_nfreq : количество частотных точек

    Возвращает:
    numpy.ndarray : массив значений I_base для каждой частоты
    """
    I_base_list = []
    for i in range(inc_nfreq):
        Ez_base = sim.get_dft_array(dtf_obj[i], num_freq=0, component=mp.Ez)
        I_base = np.sum(np.abs(Ez_base)**2)
        I_base_list.append(I_base)
    return np.array(I_base_list)

# def get_IAiry_norm_simulation_new(
#         monitor,
#         cell,
#         frequents_array: list = conf.inc_frequents_array,
#         max_run_time: int = conf.maximum_run_time,
#         load: str = None,
#         save: bool = False,
#         symmetries=None,
# ):
#     """
#     Definition:
#     Run a separate simulation per frequency and compute normalized IAiry.
#     Note. Assuming, according to diaphragm, not aperture
#
#     :param monitor: single NearRegion monitor
#     :param frequents_array: list of center frequencies
#     :param max_run_time: maximum simulation run time
#     :param load: filename to load precomputed IAiry_norm array
#     :param save: whether to save the computed result
#     :param symmetries: simulation symmetries
#     :return: IAiry_norm array shape (ff_resolution, nfreq), last sim object
#     """
#     # Load precomputed data if available
#     if load is not None:
#         IAiry_norm = np.load(load)
#         print(f"Data from {load} loaded successfully.")
#         sim = None
#         return IAiry_norm, sim
#
#     # Placeholder for normalized results
#     results = []
#     sim = None
#
#     for idx, freq in enumerate(frequents_array):
#         # Define source for this frequency
#         src = my_sources.FocusedPlaneWaveSource(
#             frequency=freq,
#             diaphragm=conf.diaphragm,
#             center=mp.Vector3(0, 0, 0),
#             focus_point=mp.Vector3(x=conf.f, y=0)
#         ).get_sources()
#
#         # Initialize simulation for this single frequency
#         sim = mp.Simulation(
#             cell_size=cell,
#             boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.ALL, side=mp.ALL)],
#             sources=[src],
#             resolution=conf.min_resolution * conf.coef_resolution,
#             symmetries=symmetries,
#         )
#
#         # Define monitors for this sim
#         n2f = sim.add_near2far(freq, 0, 1, monitor)
#         dtf = sim.add_flux(
#             freq, 0, 1,
#             mp.FluxRegion(center=conf.flux_center, size=mp.Vector3(y=cell.y))
#         )
#
#         # Run simulation
#         sim.run(until=max_run_time)
#
#         # Compute base and measured intensities
#         IAiry_base = compute_I_base(sim, [dtf], 1)
#         IAiry_meas = get_I(
#             sim,
#             [n2f],
#             conf.ff_res,
#             mp.Vector3(0, conf.spot_ff, 0),
#             out_angle_in_deg=0,
#             f=conf.f
#         )
#
#         # Normalize and store result
#         I_norm = IAiry_meas / IAiry_base
#
#         results.append(I_norm)
#
#     # Stack results to array shape (ff_res, nfreq)
#     IAiry_norm = np.stack(results, axis=1)
#
#     # Save results if requested
#     if save:
#         filename = f"Airy_simulation_D={conf.aperture}_f={conf.f}_diaph={conf.diaphragm}_res={conf.coef_resolution}_.npy"
#         np.save(filename, IAiry_norm)
#         print(f"Simulation result saved to: {filename}")
#
#     return IAiry_norm, sim

# def get_IAiry_norm_simulation_new(
#         monitor,
#         cell,
#         frequents_array: list = conf.inc_frequents_array,
#         max_run_time: int = conf.maximum_run_time,
#         symmetries=None,
#         filename: str = None,
# ):
#     """
#     Run a separate simulation per frequency and compute normalized IAiry.
#
#     :param monitor: single NearRegion monitor
#     :param frequents_array: list of center frequencies
#     :param max_run_time: maximum simulation run time
#     :param symmetries: simulation symmetries
#     :param filename: optional filename to save/load results
#     :param reload: if True, attempt to load from `filename` first
#     :return: IAiry_norm array shape (ff_resolution, nfreq), last sim object
#     """
#
#     if filename is None:
#         filename = f"Airy_simulation_D={conf.aperture}_f={conf.f}_diaph={conf.diaphragm}_res={conf.coef_resolution}.npy"
#     else:
#         if os.path.exists(filename):
#             IAiry_norm = np.load(filename)
#             print(f"Data from {filename} loaded successfully.")
#             return IAiry_norm, None
#         else:
#             ValueError(f"path does not exist: {filename}")
#
#     # Placeholder for normalized results
#     results = []
#     sim = None
#
#     for idx, freq in enumerate(frequents_array):
#         src = my_sources.FocusedPlaneWaveSource(
#             frequency=freq,
#             diaphragm=conf.diaphragm,
#             center=mp.Vector3(0, 0, 0),
#             focus_point=mp.Vector3(x=conf.f, y=0)
#         ).get_sources()
#
#         sim = mp.Simulation(
#             cell_size=cell,
#             boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.ALL, side=mp.ALL)],
#             sources=[src],
#             resolution=conf.min_resolution * conf.coef_resolution,
#             symmetries=symmetries,
#         )
#
#         n2f = sim.add_near2far(freq, 0, 1, monitor)
#         dtf = sim.add_flux(
#             freq, 0, 1,
#             mp.FluxRegion(center=conf.flux_center, size=mp.Vector3(y=cell.y))
#         )
#
#         sim.run(until=max_run_time)
#         IAiry_base = compute_I_base(sim, [dtf], 1)
#         IAiry_mean = get_I(
#             sim,
#             [n2f],
#             conf.ff_res,
#             mp.Vector3(0, conf.spot_ff, 0),
#             out_angle_in_deg=0,
#             f=conf.f
#         )
#
#         I_norm = IAiry_mean / IAiry_base
#         results.append(I_norm)
#
#     # Stack results to array shape (ff_res, nfreq)
#     IAiry_norm = np.stack(results, axis=1)
#
#     np.save(filename, IAiry_norm)
#     print(f"Simulation result saved to: {filename}")
#
#     return IAiry_norm, sim

# def get_IAiry_norm_simulation(
#         monitor,
#         cell,
#         frequents_array: list = conf.inc_frequents_array,
#         max_run_time: int = conf.maximum_run_time,
#         symmetries=None,
#         filename: str = None,
# ):
#     """
#     Run a separate simulation per frequency and compute normalized IAiry.
#
#     :param monitor: single NearRegion monitor
#     :param frequents_array: list of center frequencies
#     :param max_run_time: maximum simulation run time
#     :param symmetries: simulation symmetries
#     :param filename: optional filename to save/load results
#     :return: IAiry_norm array shape (ff_resolution, nfreq), last sim object
#     """
#     if filename is None:
#         filename = f"Airy_simulation_D={conf.aperture}_f={conf.f}_diaph={conf.diaphragm}_res={conf.coef_resolution}.npy"
#     else:
#         if os.path.exists(filename):
#             IAiry_norm = np.load(filename)
#             print(f"Data from {filename} loaded successfully.")
#             return IAiry_norm, None
#         else:
#             ValueError(f"path does not exist: {filename}")
#
#     # Placeholder for normalized results
#     base_results = []
#     results = []
#     sim = None
#
#     for idx, freq in enumerate(frequents_array):
#         src = my_sources.PlaneWaveSource(
#             frequency=freq,
#             diaphragm=conf.diaphragm,
#             center=mp.Vector3(0, 0, 0),
#         ).get_sources()
#
#         sim = mp.Simulation(
#             cell_size=cell,
#             boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.ALL, side=mp.ALL)],
#             sources=[src],
#             resolution=conf.min_resolution * conf.coef_resolution,
#             symmetries=symmetries,
#         )
#         n2f = sim.add_near2far(freq, 0, 1, monitor)
#         sim.run(until=max_run_time)
#
#         IAiry_one_wave = get_I(
#             sim,
#             [n2f],
#             conf.ff_res,
#             mp.Vector3(0, conf.spot_ff, 0),
#             out_angle_in_deg=0,
#             f=conf.f
#         )
#         IAiry_base = np.mean(IAiry_one_wave, axis=0)
#         base_results.append(IAiry_base)
#
#         src = my_sources.FocusedPlaneWaveSource(
#             frequency=freq,
#             diaphragm=conf.diaphragm,
#             center=mp.Vector3(0, 0, 0),
#             focus_point=mp.Vector3(x=conf.f, y=0)
#         ).get_sources()
#
#         sim = mp.Simulation(
#             cell_size=cell,
#             boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.ALL, side=mp.ALL)],
#             sources=[src],
#             resolution=conf.min_resolution * conf.coef_resolution,
#             symmetries=symmetries,
#         )
#
#         n2f = sim.add_near2far(freq, 0, 1, monitor)
#         sim.run(until=max_run_time)
#         IAiry_mean = get_I(
#             sim,
#             [n2f],
#             conf.ff_res,
#             mp.Vector3(0, conf.spot_ff, 0),
#             out_angle_in_deg=0,
#             f=conf.f
#         )
#         I_norm = IAiry_mean / IAiry_base
#         results.append(I_norm)
#
#     # Stack results to array shape (ff_res, nfreq)
#     IAiry_norm = np.stack(results, axis=1)
#     np.save(filename, IAiry_norm)
#     print(f"Simulation result saved to: {filename}")
#     return IAiry_norm, base_results, sim

# def get_Ibase_batch_simulation(
#         sources_batch: list,
#         cell,
#         frequents_array: list = conf.inc_frequents_array,
#         max_run_time: int = conf.maximum_run_time,
#         symmetries=None,
#         filename: str = None,
#         reload: bool = False,
# ):
#     """
#     Wrapper: run multiple simulations with different source sets
#     and return concatenated IPlane_base results only.
#     """
#
#     if filename is None:
#         filename = f"IPlane_base_D={conf.aperture}_f={conf.f}_diaph={conf.diaphragm}_res={conf.coef_resolution}.npy"
#
#     if reload and os.path.exists(filename):
#         IPlane_base_all = np.load(filename, allow_pickle=True)
#         print(f"Data from {filename} loaded successfully.")
#         return IPlane_base_all
#
#     results = []
#     n_sources = len(sources_batch)
#
#     for idx, sources in enumerate(sources_batch, start=1):
#         print(f"Running simulation {idx}/{n_sources}...")
#         IPlane_base, _ = get_Ibase_simulation(
#             sources=sources,
#             cell=cell,
#             frequents_array=frequents_array,
#             max_run_time=max_run_time,
#             symmetries=symmetries,
#         )
#         results.append(IPlane_base)
#
#     np.save(filename, results)
#     print(f"Simulation result saved to: {filename}")
#
#     return results

def get_Ibase_batch_simulation(
        monitor,
        sources_batch: list,
        cell,
        frequents_array: list = conf.inc_frequents_array,
        max_run_time: int = conf.maximum_run_time,
        symmetries=None,
        filename: str = None,
):
    """
    Wrapper: run multiple simulations with different source sets
    and return concatenated IPlane_base results only.

    Parameters:
        monitor: single NearRegion monitor
        sources_batch : list of source lists
        cell : simulation cell size
        frequents_array : list of frequencies to monitor
        max_run_time : maximum simulation time
        symmetries : optional symmetries for simulation
        filename : optional filename to save/load results
    Returns:
        IPlane_base_all : concatenated simulation results
    """

    if filename is None:
        filename = f"base_IPlane_D={conf.aperture}_f={conf.f}_diaph={conf.diaphragm}_res={conf.coef_resolution}.npy"
    else:
        if os.path.exists(filename):
            IPlane_base_all = np.load(filename)
            print(f"Data from {filename} loaded successfully.")
            return IPlane_base_all
        else:
            ValueError(f"path does not exist: {filename}")

    results = []
    n_sources = len(sources_batch)

    for idx, sources in enumerate(sources_batch, start=1):
        print(f"Running simulation {idx}/{n_sources}...")
        IPlane_base, _ = get_Ibase_simulation(
            monitor=monitor,
            sources=sources,
            cell=cell,
            frequents_array=frequents_array,
            max_run_time=max_run_time,
            symmetries=symmetries,
        )
        results.append(IPlane_base)

    # save
    np.save(filename, results)
    print(f"Simulation result saved to: {filename}")

    return results
def get_IAiry_norm_simulation(
        monitor,
        cell,
        resolution,
        frequents_array: list = conf.inc_frequents_array,
        max_run_time: int = conf.maximum_run_time,
        symmetries=None,
        filename_focus_airy: str = None,
        filename_base_airy: str = None,
        get_base: bool = False
):
    """
    Run a separate simulation per frequency and compute normalized IAiry.

    :param monitor: single NearRegion monitor
    :param frequents_array: list of center frequencies
    :param max_run_time: maximum simulation run time
    :param symmetries: simulation symmetries
    :param filename: optional filename to save/load results
    :return: IAiry_norm array shape (ff_resolution, nfreq), last sim object
    """
    if filename_focus_airy is None:
        filename_focus_airy = f"focus_Airy_simulation_D={conf.aperture}_f={conf.f}_diaph={conf.diaphragm}_res={conf.coef_resolution}.npy"
        filename_base_airy = f"base_Airy_simulation_D={conf.aperture}_f={conf.f}_diaph={conf.diaphragm}_res={conf.coef_resolution}.npy"
    else:
        IAiry_focus = None
        IAiry_base = None

        if os.path.exists(filename_focus_airy):
            IAiry_focus = np.load(filename_focus_airy)
            print(f"Data from {filename_focus_airy} loaded successfully.")
        else:
            ValueError(
                f"path does not exist: {filename_focus_airy}")

        if os.path.exists(filename_base_airy):
            IAiry_base = np.load(filename_base_airy, allow_pickle=True)
            print(f"Data from {filename_base_airy} loaded successfully.")

        return IAiry_focus, IAiry_base, None

        # Placeholder for normalized results
    base_results = []
    results = []
    sim = None

    for idx, freq in enumerate(frequents_array):
        if get_base:
            if len(base_results) == 0:
                src = my_sources.EigenmodeSource(
                    frequency=freq,
                    diaphragm=conf.diaphragm,
                    center=mp.Vector3(0,0,0),
                    resolution=resolution,
                    angle_in_rad=0,
                ).get_sources()

                sim = mp.Simulation(
                    cell_size=cell,
                    boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.ALL, side=mp.ALL)],
                    sources=[src],
                    resolution=conf.min_resolution * conf.coef_resolution,
                    symmetries=symmetries,
                )
                n2f = sim.add_near2far(freq, conf.inc_df, 1, monitor)
                sim.run(until=max_run_time)

                IAiry_base_single_wave = get_I(
                    sim,
                    [n2f],
                    conf.ff_res,
                    mp.Vector3(0, conf.spot_ff, 0),
                    out_angle_in_deg=0,
                    f=conf.f
                )
                base_results.append(IAiry_base_single_wave)

        src = my_sources.FocusedEigenmodeSource(
            frequency=freq,
            diaphragm=conf.diaphragm,
            center=mp.Vector3(0,0,0),
            focus_point=mp.Vector3(x=conf.f, y=0)
        ).get_sources()

        sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.ALL, side=mp.ALL)],
            sources=[src],
            resolution=conf.min_resolution * conf.coef_resolution,
            symmetries=symmetries,
        )

        n2f = sim.add_near2far(freq, 0, 1, monitor)
        sim.run(until=max_run_time)
        IAiry_focus_single_wave = get_I(
            sim,
            [n2f],
            conf.ff_res,
            mp.Vector3(0, conf.spot_ff, 0),
            out_angle_in_deg=0,
            f=conf.f
        )
        results.append(IAiry_focus_single_wave)

    # Stack results to array shape (ff_res, nfreq)
    IAiry_focus = np.stack(results, axis=1)
    if get_base:
        IAiry_base = np.stack(base_results, axis=1)
    else:
        IAiry_base = None
    np.save(filename_focus_airy, IAiry_focus)
    print(f"Simulation focus result saved to: {filename_focus_airy}")
    np.save(filename_base_airy, IAiry_base)
    print(f"Simulation base result saved to: {filename_base_airy}")
    return IAiry_focus, IAiry_base, sim

# def get_Ibase_simulation(
#         monitor,
#         sources: list,
#         cell,
#         frequents_array: list = conf.inc_frequents_array,
#         max_run_time: int = conf.maximum_run_time,
#         symmetries = None,
#
# ):
#     """
#     Definition: use function to get I_base simulated bunch waves for further normalization
#     :param monitor: single NearRegion monitor
#     :param angle:
#     :param frequents_array:
#     :param max_run_time:
#     :return:
#     """
#     results = []
#     sim = None
#
#     geometry = [mp.Block(
#         material=conf.in_material,
#         size=mp.Vector3(conf.in_sub, mp.inf, 0),
#         center=mp.Vector3(conf.sub_center.x),
#     )]
#
#     for idx, freq in enumerate(frequents_array):
#         sim = mp.Simulation(
#             cell_size=cell,
#             boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.X, side=mp.ALL)],
#             sources=sources,
#             geometry=geometry,
#             resolution=conf.min_resolution * conf.coef_resolution,
#             symmetries = symmetries, # При наклонном падении света симметрию использовать нельзя
#         )
#
#         # add monitors
#         n2f = sim.add_near2far(freq, 0, 1, monitor)
#         sim.run(until=max_run_time)
#
#         IBase_one_wave = get_I(
#             sim,
#             [n2f],
#             conf.ff_res,
#             mp.Vector3(0, conf.spot_ff, 0),
#             out_angle_in_deg=0,
#             f=conf.f
#         )
#         results.append(IBase_one_wave)
#
#     return results, sim

def get_Ibase_simulation(
        monitor,
        sources: list,
        cell,
        frequents_array: list = conf.inc_frequents_array,
        max_run_time: int = conf.maximum_run_time,
        symmetries = None,

):
    """
    Definition: use function to get I_base simulated bunch waves for further normalization
    :param monitor: single NearRegion monitor
    :param angle:
    :param frequents_array:
    :param max_run_time:
    :return:
    """

    geometry = [mp.Block(
        material=conf.in_material,
        size=mp.Vector3(conf.in_sub, mp.inf, 0),
        center=mp.Vector3(conf.sub_center.x),
    )]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.X, side=mp.ALL)],
        sources=sources,
        geometry=geometry,
        resolution=conf.min_resolution * conf.coef_resolution,
        symmetries=symmetries,  # При наклонном падении света симметрию использовать нельзя
    )

    near2far_regions = []
    for freq in frequents_array:
        near2far_regions.append(
            sim.add_near2far(
                freq,           # центральная частота
                conf.inc_df,    # ширина спектра
                1,              # дискретность
                monitor
            )
        )
    sim.run(until=max_run_time)
    IBase_all_wave = get_I(
        sim,
        near2far_regions,
        conf.ff_res,
        mp.Vector3(0, conf.spot_ff, 0),
        out_angle_in_deg=0,
        f=conf.f
    )
    return [IBase_all_wave], sim

def get_I(sim, near2far_region, ff_res, size, out_angle_in_deg, f):
    """
    :param sim: simulation object
    :param near2far_region: single near2far region or list of near2far regions
    :param ff_res: коэффициент разрешения для дальнего поля
    :param size: mp.Vector3 с ненулевым значением по интересующей координате
    :param out_angle_in_deg: угол для вычисления смещения
    :param f: фокусное расстояние
    :return:
        Если near2far_region - список: 2D массив shape (n, m),
            где n - размер результата для одного региона, m - количество регионов
        Если одиночный регион: 1D массив shape (n,)
    """
    def calculate_single_region(region):
        far_field = sim.get_farfields(
            region,
            ff_res,
            center=mp.Vector3(x=f, y=shift_y(angle=out_angle_in_deg)),
            size=size
        )
        return abs(far_field["Ez"]) ** 2

    if isinstance(near2far_region, (list, tuple)):
        # Если передан список регионов, вычисляем для каждого и собираем в columns
        I_list = [calculate_single_region(region) for region in near2far_region]
        I_arr = np.column_stack(I_list) if len(I_list) > 1 else I_list[0]
    else:
        # Если передан одиночный регион
        I_arr = calculate_single_region(near2far_region)

    return I_arr

def get_I_loc(sim, near2far_region, shift, ff_res, size, f):
    """
    :param sim: simulation object
    :param near2far_region: single near2far region or list of near2far regions
    :param ff_res: коэффициент разрешения для дальнего поля
    :param size: mp.Vector3 с ненулевым значением по интересующей координате
    :param out_angle_in_deg: угол для вычисления смещения
    :param f: фокусное расстояние
    :return:
        Если near2far_region - список: 2D массив shape (n, m),
            где n - размер результата для одного региона, m - количество регионов
        Если одиночный регион: 1D массив shape (n,)
    """

    def calculate_single_region(region):
        far_field = sim.get_farfields(
            region,
            ff_res,
            center=mp.Vector3(x=f, y=shift),
            size=size
        )
        return abs(far_field["Ez"]) ** 2

    if isinstance(near2far_region, (list, tuple)):
        # Если передан список регионов, вычисляем для каждого и собираем в columns
        I_list = [calculate_single_region(region) for region in near2far_region]
        I_arr = np.column_stack(I_list) if len(I_list) > 1 else I_list[0]
    else:
        # Если передан одиночный регион
        I_arr = calculate_single_region(near2far_region)

    return I_arr

def get_indx_orders(index_angle):
    if index_angle == 0:
        global_start_indx = 0
        global_end_indx = (conf.diaphragm / 2) // conf.order_step
    else:
        global_start_indx = 1 + (conf.diaphragm / 2 + conf.diaphragm * (index_angle - 1)) // conf.order_step
        global_end_indx = (conf.diaphragm / 2 + conf.diaphragm * index_angle) // conf.order_step

    return int(global_start_indx), int(global_end_indx)

"""
PLOT
"""

def get_plot_intensity_multiply(spot_ff, I_ff_list, angle):
    """
    :param spot_ff: размер области расчета
    :param I_ff_list: список массивов интенсивности (каждый нормализован от 0 до 1),
                      либо один массив (не в списке)
    :param angle: угол падения
    """
    # Убедимся, что вход всегда список массивов
    if not isinstance(I_ff_list, list):
        I_ff_list = [I_ff_list]

    plt.figure(dpi=100)

    # Отображение каждой кривой
    for idx, I_ff in enumerate(I_ff_list):
        abscissa = get_abscissa(spot_ff, I_ff, angle)
        label = f"Intensity {idx + 1}" if len(I_ff_list) > 1 else f"Far field ({conf.inc_wavelength} μm)"
        plt.plot(abscissa, I_ff, label=label)

    plt.xlabel(f"x coordinate in µm")
    plt.ylabel(r"Normalized energy density, |E$_z$|$^2$ (0...1)")
    plt.title(f"Intensity of CC lens. Incident {angle}°")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def get_plot_intensity(I_ff, out_angle_in_deg, wavelength_array, abscissa, isfwhm = True):
    """
    :param spot_ff: размер области расчета
    :param I_ff: значения интенсивностей (нормализованные от 0 до 1)
    :param out_angle_in_deg: угол до преломления
    :param is_loc: локальная оптимизация или глобальные

    """
    plt.style.use('default')
    plt.figure(dpi=100)

    if len(wavelength_array) == 1:
        plt.plot(abscissa, I_ff, label=f"Far field ({conf.inc_wavelength} μm)")
    else:
        for i in range(I_ff.shape[1]):
            plt.plot(abscissa, I_ff[:, i], label=f"Far field (wavelength {wavelength_array[i]})")

    plt.xlabel(f"x coordinate [µm]")
    plt.ylabel(r"Normalized energy density, |E$_z$|$^2$ (0...1)")
    plt.title(f"Intensity of CC lens. Out Incident {np.around(out_angle_in_deg, decimals = 1 )} degree")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    if isfwhm:
        if len(wavelength_array) == 1:
            fwhm = get_fwhm(abscissa, I_ff)
            print(f"FWHM = {fwhm:.2f} µm")
        else:
            for i in range(I_ff.shape[1]):
                fwhm = get_fwhm(abscissa, I_ff[:, i])
                print(f"λ={wavelength_array[i]}: FWHM = {fwhm:.2f} µm")

def get_fwhm(abscissa, intensity):
    """
    Вычисление FWHM (на уровне 0.5) и FWTM (на уровне 0.1)
    для одномерного профиля интенсивности.

    :param abscissa: массив координат (например, [мкм])
    :param intensity: массив нормализованных интенсивностей (0...1)
    :return: (FWHM, x_left_half, x_right_half, FWTM, x_left_tenth, x_right_tenth)
    """
    center = np.argmax(intensity)
    n = len(intensity)

    def find_width(level):
        """Вспомогательная функция поиска ширины на уровне level."""
        left_idx, right_idx = None, None

        # поиск влево
        for i in range(center, -1, -1):
            if intensity[i] <= level:
                left_idx = i
                break
        # поиск вправо
        for i in range(center, n):
            if intensity[i] <= level:
                right_idx = i
                break

        if left_idx is None or right_idx is None:
            raise ValueError(f"Не удалось найти два пересечения на уровне {level}.")

        x_left = float(abscissa[left_idx])
        x_right = float(abscissa[right_idx])
        width = x_right - x_left
        return width, x_left, x_right

    # FWHM (0.5)
    fwhm, x_left_half, x_right_half = find_width(0.5)


    return fwhm

def get_focusing_efficiency(IAiry_norm, abscissa):
    """
    Вычисляет эффективность фокусировки как долю энергии
    в интервале ширины airy_diameter относительно центра профиля.

    Parameters
    ----------
    IAiry_norm : array_like
        Нормализованный (на максимум) вектор или матрица интенсивностей.
        - Если размерность (n,), то n точек профиля для одной длины волны.
        - Если размерность (n, m), то m разных длин волн, каждая колонка — отдельный профиль.
    spot_ff, out_angle_in_deg, shift : параметры для get_abscissa
        Используются для генерации координат абсцисс.

    Returns
    -------
    eta : float или ndarray
        Эффективность фокусировки (0..1).
        - float, если m=1
        - ndarray формы (m,), если m>1
    """
    IAiry_norm = np.atleast_2d(IAiry_norm)
    if IAiry_norm.shape[0] == 1:  # приведение к (n, m), а не (1, n)
        IAiry_norm = IAiry_norm.T

    n, m = IAiry_norm.shape

    # получаем ось абсцисс (одна для всех длин волн)
    dx = np.mean(np.diff(abscissa))

    # центр диапазона координат
    center = 0.5 * (abscissa[0] + abscissa[-1])

    # поддержка скаляра и вектора airy_diameter
    airy_diam = np.atleast_1d(conf.airy_diameter)
    if airy_diam.size == 1 and m > 1:
        airy_diam = np.full(m, airy_diam.item())
    elif airy_diam.size != m:
        raise ValueError("conf.airy_diameter должен быть скаляром или массивом длины m")

    eta = np.zeros(m)
    for j in range(m):
        mask = (abscissa >= center - airy_diam[j] / 2) & (abscissa <= center + airy_diam[j] / 2)
        power_in = np.sum(IAiry_norm[mask, j]) * dx
        power_total = np.sum(IAiry_norm[:, j]) * dx
        eta[j] = power_in / power_total if power_total > 0 else 0.0

    return eta[0] if m == 1 else eta

def plot_psf(
    nfrec: int,
    spot_ff: Union[float, Sequence[float]],
    I: np.ndarray,
    abscissa,
) -> None:
    """
    Строит и отображает PSF (Point Spread Function) на основе одномерного или двумерного
    распределения интенсивности. Если передана `abscissa`, она будет использована в качестве
    оси x, иначе она будет рассчитана автоматически.

    Parameters
    ----------
    nfrec : int
        Число подграфиков (дискретность длин волн).
    spot_ff : float или array_like
        Пространственная область (например, апертура), на которой заданы интенсивности.
    I : ndarray
        Массив интенсивностей:
        - если форма (M,), то будет построен один PSF;
        - если форма (M, N), то будет построено N PSF.

    Raises
    ------
    ValueError
        Если `I` имеет размерность, отличную от 1 или 2.
    """

    # Проверка формы входных данных
    I = np.asarray(I)
    if I.ndim == 1:
        # Преобразуем к двумерному виду (M, 1)
        I = I[:, np.newaxis]
    elif I.ndim != 2:
        raise ValueError(f"Ожидается 1D или 2D массив, получили массив с ndim={I.ndim}")

    # Выбираем абсциссу: пользовательскую или автоматически вычисленную
    x_plot = abscissa
    x_calculate = get_abscissa(spot_ff, I[:, 0], 0)

    # Создаём сетку для интерполяции
    X, Y = np.meshgrid(x_calculate, x_calculate)
    R = np.hypot(X, Y)  # Более «профессиональный» способ sqrt(X**2+Y**2)

    # Подготовка фигуры
    fig, axes = plt.subplots(1, nfrec, figsize=(6 * nfrec, 6), squeeze=False)
    axes = axes.ravel()

    wavelength_array = conf.inc_wavelength_array
    for idx in range(nfrec):
        if idx >= I.shape[1]:
            # Если колонок меньше, чем nfrec, заполняем нулями
            intensity = np.zeros_like(x_calculate)
        else:
            intensity = I[:, idx]

        # Интерполяция и нормализация
        interp_func = interp1d(x_calculate, intensity, bounds_error=False, fill_value=0.0)
        I2d = interp_func(R)
        I2d -= I2d.min()
        if I2d.max() > 0:
            I2d /= I2d.max()

        # Отображение
        im = axes[idx].imshow(
            I2d,
            extent=(x_plot[0], x_plot[-1], x_calculate[0], x_calculate[-1]),
            origin="lower",
            cmap="viridis",
            # cmap = "hot",
            aspect="equal",
        )


        if len(wavelength_array) == 1:
            axes[idx].set(
                title=f"PSF for {conf.inc_wavelength}",
                xlabel="Spot [μm]",
                ylabel="Spot [μm]",
            )
        else:
            axes[idx].set(
                title=f"PSF for {wavelength_array[idx]}",
                xlabel="Spot [μm]",
                ylabel="Spot [μm]",
            )
        fig.colorbar(im, ax=axes[idx], label="Normalized intensity")

    plt.tight_layout()
    fig.suptitle(f"Resolution: {len(x_plot)} pixels", fontsize=16, color="blue", y=1.02)
    plt.show()

# def plot_mtf(
#         spot_ff: Union[float, Sequence[float]],
#         I_ff: np.ndarray,
#         in_angle_in_deg: float,
#         out_angle_in_deg: float,
#         max_view: float = 50,
#         dif_lim=False,
#         smooth=False,
# ) -> None:
#     """
#     Строит кривую MTF на основе распределения интенсивности в фокальной плоскости.
#
#     Parameters
#     ----------
#     spot_ff : float или array_like
#         Пространственная область (например, апертура).
#     I_ff : ndarray
#         Массив интенсивностей PSF (1D или 2D).
#     dif_lim : bool
#         Вкл значения для идеального случая
#     """
#
#     # Проверка и подготовка данных
#     I_ff = np.asarray(I_ff)
#     if I_ff.ndim == 1:
#         I_ff = I_ff[:, np.newaxis]
#
#     # Получение пространственных координат
#     x = get_abscissa(spot_ff, I_ff[:, 0], angle = 0)
#     dx = np.abs(x[1] - x[0])  # Шаг дискретизации в микронах
#     dx_mm = dx * 1e-3  # Шаг дискретизации в mm
#     print(f"len of x = {len(x)}")
#
#     # Расчет частотной сетки
#     N = len(x)
#     freq = np.fft.fftfreq(N, d=dx_mm)[:N // 2 + 1]  # Берем первую половину. [1/μm]
#     # Подготовка графика
#     wavelength_array = conf.inc_wavelength_array
#
#     for i in range(I_ff.shape[1]):
#         # Нормализация PSF
#         psf = I_ff[:, i]
#         psf /= np.sum(psf)
#
#         # Расчет MTF
#         mtf = np.abs(fft(psf))[:N // 2 + 1]
#         mtf_normalized = mtf / mtf.max()
#
#         # Построение кривой
#         label = (
#             f"MTF ({wavelength_array[i]} μm)"
#             if len(wavelength_array) > 1
#             else "MTF"
#         )
#
#         if smooth:
#             # Применяем фильтр к рассчитанному MTF
#             window_length = 7  # Должно быть нечетным и меньше размера данных
#             polyorder = 2
#             mtf_normalized = savgol_filter(mtf_normalized, window_length, polyorder)
#             print(f"smooth x = {len(mtf_normalized)}")
#
#         plt.plot(freq, mtf_normalized, label=label)
#
#     theta_rad = np.radians(in_angle_in_deg)
#
#     if dif_lim:
#         # если у нас несколько длин волн — работаем по каждой
#         for lam in wavelength_array:
#             # переводим λ из мкм в мм
#             lam_mm = lam * 1e-3
#             # предельная частота в lp/mm
#             fc = 1 / lam_mm
#             # коррекция частоты
#             fc_eff = fc * np.cos(theta_rad)
#             # задаём сетку частот от 0 до fc_eff
#             f_ratio = np.linspace(0, fc_eff, 200)
#             # считаем теоретическую MTF
#             mtf_th = (2 / np.pi) * (
#                 np.arccos(f_ratio / fc_eff)
#                 - (f_ratio / fc_eff) * np.sqrt(1 - (f_ratio / fc_eff) ** 2)
#             )
#             # строим штриховую линию для каждой λ
#             plt.plot(f_ratio, mtf_th, linestyle='--', label=f'Diffraction limit {lam} µm')
#
#     # Настройка осей
#     plt.xlim(0, max_view)  # Только положительные частоты
#     plt.xlabel("Spatial Frequency (lp/mm)")
#     plt.ylabel("Contrast (MTF)")
#     plt.title(f"Modulation Transfer Function for {np.around(out_angle_in_deg, decimals= 1)} deg")
#     plt.ylim(0, 1.1)
#     plt.grid(False)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

def plot_mtf(
        spot_ff: Union[float, Sequence[float]],
        I_ff: np.ndarray,
        in_angle_in_deg: float,
        out_angle_in_deg: float,
        max_view: float = 50,
        smooth: bool = False,
        dif_lim: bool = False,
        I_mtf_theory: np.ndarray = None,
) -> None:
    """
    Строит кривую MTF на основе распределения интенсивности в фокальной плоскости.

    Parameters
    ----------
    spot_ff : float или array_like
        Пространственная область (например, апертура).
    I_ff : ndarray
        Массив интенсивностей PSF (1D или 2D).
    in_angle_in_deg : float
        Угол на входе.
    out_angle_in_deg : float
        Угол на выходе.
    max_view : float
        Максимальная частота на графике (lp/mm).
    dif_lim : bool
        Включать значения для идеального случая.
    smooth : bool
        Сглаживание рассчитанных MTF.
    I_mtf_theory : ndarray
        Предварительно рассчитанные значения идеальной MTF.
        Если переданы, то расчет дифракционного предела внутри функции не выполняется.
    """

    # Проверка и подготовка данных
    I_ff = np.asarray(I_ff)
    if I_ff.ndim == 1:
        I_ff = I_ff[:, np.newaxis]

    # Получение пространственных координат
    x = get_abscissa(spot_ff, I_ff[:, 0], angle=0)
    dx = np.abs(x[1] - x[0])  # Шаг дискретизации в микронах
    dx_mm = dx * 1e-3  # Шаг дискретизации в mm
    print(f"len of x = {len(x)}")

    # Расчет частотной сетки
    N = len(x)
    freq = np.fft.fftfreq(N, d=dx_mm)[:N // 2 + 1]  # Берем первую половину. [1/μm]
    # Подготовка графика
    wavelength_array = conf.inc_wavelength_array

    theta_rad = np.radians(in_angle_in_deg)

    for i in range(I_ff.shape[1]):
        # Нормализация PSF
        psf = I_ff[:, i]
        psf /= np.sum(psf)

        # Расчет MTF
        mtf = np.abs(fft(psf))[:N // 2 + 1]
        mtf_normalized = mtf / mtf.max()

        # Построение кривой
        label = (
            f"optimized ({wavelength_array[i]} μm)"
        )

        if smooth:
            # Применяем фильтр к рассчитанному MTF
            window_length = 7  # Должно быть нечетным и меньше размера данных
            polyorder = 2
            mtf_normalized = savgol_filter(mtf_normalized, window_length, polyorder)
            print(f"smooth x = {len(mtf_normalized)}")

        plt.plot(freq, mtf_normalized, label=label)

    if dif_lim:
        if I_mtf_theory is not None:
            # Используем готовые значения
            for i in range(I_mtf_theory.shape[1]):
                # Нормализация PSF
                psf = I_mtf_theory[:, i]
                psf /= np.sum(psf)

                # Расчет MTF
                mtf = np.abs(fft(psf))[:N // 2 + 1]
                mtf_normalized = mtf / mtf.max()

                # Построение кривой
                label = (
                    f"limit_sim ({wavelength_array[i]} μm)"
                )
                if smooth:
                    # Применяем фильтр
                    window_length = 7  # Должно быть нечетным и меньше размера данных
                    polyorder = 2
                    mtf_normalized = savgol_filter(mtf_normalized, window_length, polyorder)
                    print(f"smooth x = {len(mtf_normalized)}")

                plt.plot(freq, mtf_normalized, linestyle='--', label=label)
        else:
            for lam in wavelength_array:
                lam_mm = lam * 1e-3  # переводим λ из мкм в мм
                fc = 1 / lam_mm
                fc_eff = fc * np.cos(theta_rad)
                f_ratio = np.linspace(0, fc_eff, 200)
                mtf_th = (2 / np.pi) * (
                    np.arccos(f_ratio / fc_eff)
                    - (f_ratio / fc_eff) * np.sqrt(1 - (f_ratio / fc_eff) ** 2)
                )
                plt.plot(
                    f_ratio, mtf_th,
                    linestyle='--',
                    label=f'limit_theo {lam} µm'
                )

    # Настройка осей
    plt.xlim(0, max_view)  # Только положительные частоты
    plt.xlabel("Spatial Frequency (lp/mm)")
    plt.ylabel("Contrast (MTF)")
    plt.title(f"Modulation Transfer Function for {np.around(out_angle_in_deg, decimals=1)} deg")
    plt.ylim(0, 1.1)
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.show()

def setup_visualization():
    """Инициализация фигуры для визуализации"""
    global fig, ax
    fig, ax = plt.subplots(figsize=(10, 10))
    # Определяем палитру цветов
    global colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_curve(xx, yy, X_sections, y0, n, is_cut = True):
    """
    Функция для визуализации кривой и точек пересечения
    :param xx: x-координаты кривой
    :param yy: y-координаты кривой
    :param X_sections: точки пересечения с горизонтальной линией
    :param y0: y-координата горизонтальной линии
    :param n: номер кривой (для выбора цвета)
    """

    # Рисуем кривую
    ax.plot(xx, yy, color=colors[n % len(colors)], linewidth=0.5)

    # Выделяем точки пересечения красным
    if is_cut:
        ax.scatter(X_sections, np.full_like(X_sections, y0), color='red', s=10, zorder=3)


def show_lens(
        a_params: np.ndarray,  # shape = (orders,)
        orders: int,
        rt: np.ndarray,        # shape = (orders,)
        Q: int = 200,
        w: float = 0.1,        # линейное расстояние между ячейками
        plot_size: int = 20,
    ):
    """
    Визуализация равномерного окружного размещения ячеек-линз.

    Каждый сегмент m отображается профилем y = a * cosh((x - x0)/a),
    обрезанным по высоте и уложенным по кольцу радиуса r = (rt0 + rt1)/2.
    rt0 и rt1 — нижняя и верхняя границы радиуса сегмента, без дополнительных смещений.
    Диаметр всей линзы = 2 * rt[-1].
    """
    fig, ax = plt.subplots(figsize=(plot_size, plot_size))

    for m in range(orders):
        a = max(a_params[m], 1e-3)

        # Границы сегмента по радиусу (без +0.01/-0.01)
        rt0 = 0.0 if m == 0 else rt[m - 1] + 0.01
        rt1 = rt[m] - 0.01

        # Если верхняя <= нижней — пропускаем
        if rt1 <= rt0:
            continue

        # Центр профиля по x и радиус кольца
        x0 = 0.0 if m == 0 else (rt[m] + rt0) / 2
        r  = x0

        # Диапазон по x, «надрез» профиля
        L = np.clip(1.5 / a, 2.0, 8.0)
        x_left  = max(rt0, x0 - L * a)
        x_right = min(rt1, x0 + L * a)
        if x_right <= x_left:
            continue

        x = np.linspace(x_left, x_right, Q)
        z = np.clip((x - x0) / a, -20, 20)
        y_profile = a * np.cosh(z)

        # Сдвиг так, чтобы минимум профиля был в нуле
        ip = np.argmin(y_profile)
        y_shifted = y_profile - y_profile[ip]

        # Обрезка «по высоте» для визуализации
        ymax = 0.5 * (x.max() - x.min())
        y_top = np.clip(y_shifted, -ymax, ymax)
        y_bot = np.clip(y_shifted - w, -ymax, ymax)

        xx = np.concatenate([x, x[::-1]])
        yy = np.concatenate([y_top, y_bot[::-1]])

        # Число ячеек на кольце
        if r < 1e-6:
            continue
        w_rad = (w / r)
        N = max(3, int(np.floor(2 * np.pi / w_rad)))

        # Рисуем сегменты по окружности
        for n in range(N):
            theta = 2 * np.pi * n / N
            xx_local = xx - x0
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            x_rot = xx_local * cos_t - yy * sin_t + r * cos_t
            y_rot = xx_local * sin_t + yy * cos_t + r * sin_t
            ax.plot(x_rot, y_rot, color='blue', linewidth=0.5)

    ax.set_aspect('equal')
    ax.set_title("Равномерное окружное размещение ячеек")
    plt.show()

def generate_lens_gds_2params(
        a_params,
        h_params,
        orders: int,
        rt,
        n_layers: int = 1,
        w: float = 0.1,
        Q: int = 100,
        filename: str = 'lens_layout.gds',
        cell_name: str = 'LENS_CELL',
        unit: float = 1e-6,
        precision: float = 20e-9
) -> None:
    """
    Генерирует многослойный GDSII файл.
    Высота дискретизируется шагом 100 нм (0.1 мкм).
    Каждый порядок присутствует во всех слоях до своей высоты h_params[m].
    Добавляется отдельная ячейка LEGEND с подписями слоев.
    """
    # На случай, если приходят тензоры torch
    a_params = np.asarray(a_params, dtype=float)
    h_params = np.asarray(h_params, dtype=float)
    rt = np.asarray(rt, dtype=float)

    # Создаем библиотеку и главную ячейку
    lib = gdspy.GdsLibrary(unit=unit, precision=precision)
    cell = lib.new_cell(cell_name)

    # Количество слоев (0–30 включительно => 30 слой)
    n_layers = n_layers
    layer_height = 0.1  # мкм

    for layer in range(n_layers):
        current_height = layer * layer_height  # мкм

        for m in range(orders):
            a = max(a_params[m], 1e-3)
            h = h_params[m]

            if h < current_height - 1e-12:
                continue

            rt0 = 0.0 if m == 0 else rt[m - 1] + 0.1
            rt1 = rt[m] - 0.1
            x0 = 0.0 if m == 0 else (rt[m] + rt0) / 2
            r = x0
            if r < 1e-6:
                continue

            L = np.clip(1.5 / a, 2.0, 10.0)
            x_left = max(rt0, x0 - L * a)
            x_right = min(rt1, x0 + L * a)
            if x_right <= x_left:
                continue

            x = np.linspace(x_left, x_right, Q)
            z = np.clip((x - x0) / a, -20, 20)
            y_profile = a * np.cosh(z)

            ip = np.argmin(y_profile)
            y_shifted = y_profile - y_profile[ip]

            ymax = 0.5 * (x.max() - x.min())
            y_top = np.clip(y_shifted, -ymax, ymax)
            y_bot = np.clip(y_shifted - w, -ymax, ymax)

            xx = np.concatenate([x, x[::-1]])
            yy = np.concatenate([y_top, y_bot[::-1]])

            w_rad_cell = w / r
            w_rad_gap = w / r
            T_rad = w_rad_cell + w_rad_gap
            N = max(1, int(np.floor(2 * np.pi / T_rad)))

            for n in range(N):
                theta0 = n * T_rad
                cos0 = np.cos(theta0)
                sin0 = np.sin(theta0)
                xx_local = xx - x0
                coords = np.vstack([
                    xx_local * cos0 - yy * sin0 + r * cos0,
                    xx_local * sin0 + yy * cos0 + r * sin0
                ]).T
                cell.add(gdspy.Polygon(coords, layer=layer))

    # Сохраняем файл
    lib.write_gds(filename)
    print(f"GDS файл успешно сохранён: {filename}")

# works
def generate_lens_stl_linear(
    a_params,
    h_params,
    orders: int,
    rt,
    N: int = 10,
    w: float = 0.1,
    Q: int = 100,
    substrate_thickness: float = 10.0,
    filename: str = "lens_layout_linear.stl"
) -> None:
    """
    Генерирует STL-модель линзы на подложке.
    Ячейки размещаются линейно (±N копий по оси Y) и зеркально по оси X.
    Все размерности в микронах.
    """
    a_params = np.asarray(a_params, dtype=float)
    h_params = np.asarray(h_params, dtype=float)
    rt = np.asarray(rt, dtype=float)
    substrate_thickness = substrate_thickness

    assert len(a_params) == orders
    assert len(h_params) == orders
    assert len(rt) == orders

    meshes = []

    # ---- Подложка ----
    W = (N + 1) * w * 8
    L = rt[-1] * 2
    H = substrate_thickness
    substrate = trimesh.creation.box(extents=(L, W, H))
    substrate.apply_translation((0, 0, H / 2))
    meshes.append(substrate)

    # ---- Сегменты линзы ----
    for m in range(orders):
        a = max(a_params[m], 1e-3)
        h = h_params[m]

        rt0 = 0.0 if m == 0 else rt[m - 1] + 0.1
        rt1 = rt[m]
        x0 = 0.0 if m == 0 else (rt[m] + rt0) / 2

        L_seg = np.clip(1.5 / a, 2.0, 10.0)
        x_left = max(rt0, x0 - L_seg * a)
        x_right = min(rt1, x0 + L_seg * a)
        if x_right <= x_left:
            continue

        x = np.linspace(x_left, x_right, Q)
        z = np.clip((x - x0) / a, -20, 20)
        y_profile = a * np.cosh(z)

        ip = np.argmin(y_profile)
        y_shifted = y_profile - y_profile[ip]

        ymax = 0.5 * (x.max() - x.min())
        y_top = np.clip(y_shifted, -ymax, ymax)
        y_bot = np.clip(y_shifted - w, -ymax, ymax)

        xx = np.concatenate([x, x[::-1]])
        yy = np.concatenate([y_top, y_bot[::-1]])

        # ---- Генерация копий вдоль Y и зеркальные по X ----
        y_offsets = np.arange(-N, N+1) * w * 2
        x_signs = [1, -1]  # зеркальная симметрия по X

        for x_sign in x_signs:
            for y_off in y_offsets:
                coords = np.vstack([xx * x_sign, yy + y_off]).T
                poly = Polygon(coords)

                # Очистка самопересечений и ориентация
                if not poly.is_valid:
                    try:
                        poly = poly.buffer(0)
                        poly = orient(poly)
                        if not poly.is_valid:
                            from shapely.validation import explain_validity
                            print(f"⚠ После buffer(0) полигон всё ещё невалиден для сегмента {m}, смещение {y_off}")
                            print(f"  Причина: {explain_validity(poly)}")
                            continue
                    except Exception as e:
                        print(f"Ошибка при очистке полигона для сегмента {m}, смещение {y_off}: {e}")
                        continue

                if poly.area < 1e-6:
                    print(f"⚠ Малая площадь полигона ({poly.area:.6f}) для сегмента {m}, смещение {y_off}")

                # Экструзия на высоту
                try:
                    tri = trimesh.creation.extrude_polygon(poly, h, engine="triangle")
                except Exception:
                    tri = trimesh.creation.extrude_polygon(poly, h)

                tri.apply_translation((0, 0, substrate_thickness))
                meshes.append(tri)

    # ---- Итоговый меш ----
    if meshes:
        final_mesh = trimesh.util.concatenate(meshes)
        final_mesh.export(filename)
        print(f"STL сохранён: {filename}")
    else:
        print("⚠ Нет геометрии для экспорта в STL")


def airy_characteristics():
    """
    Вычисляет характерные размеры для идеального фокусного пятна (Airy disk).

    :param NA: числовая апертура
    :param wavelength: длина волны (в тех же единицах, например, мкм)
    :return: dict с FWHM, FWTM, диаметром первого минимума
    """
    fwhm = 0.514 * conf.inc_wavelength / conf.numerical_aperture
    fwtm = 0.885 * conf.inc_wavelength / conf.numerical_aperture
    airy_diameter = 1.22 * conf.inc_wavelength / conf.numerical_aperture
    print(f"FWHM: {fwhm:.2f} um , FWTM: {fwtm:.2f} um, Airy diameter: {airy_diameter:.2f} um")
