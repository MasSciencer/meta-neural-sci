import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import os
from scipy.interpolate import interp1d
from scipy.fft import fft
from scipy.signal import savgol_filter
from typing import Union, Sequence,List, Iterable, Tuple, Optional
import gdspy
from configs import conf_test as conf
from lib import my_sources
import trimesh
from shapely.geometry import Polygon
from shapely.ops import orient

class AdaptiveResolution:
    def __init__(
        self,
        drt: Iterable[float],
        x_dim: int,
        h_dim: int,
    ):
        """
        :param drt: ширины сегментов по x
        :param x_dim: фиксированное разрешение по x
        :param h_dim: разрешение по y
        """
        self.drt = list(drt)
        self.x_dim = x_dim
        self.height_dim = h_dim

    def generate(self) -> List[Tuple[int, int]]:
        """Возвращает список разрешений в формате [(w, h), ...]"""
        return [(self.x_dim, self.height_dim) for _ in self.drt]

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
        numpy радиусов (float64), упорядоченный по возрастанию,
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

def get_loc_orders(diaphragm_orders, is_center):
    """
    :param diaphragm_orders:
    :param is_center:
    :return: loc_orders
    """
    if is_center:
        loc_orders = diaphragm_orders / 2
    else:
        loc_orders = diaphragm_orders

    return int(loc_orders)

def get_drt_midrt(rt, orders, is_center=True):
    """
    :param rt: numpy array с радиусами
    :param orders: количество порядков
    :param is_center: режим вычисления
    :return: drt (np.ndarray), midrt (np.ndarray)
    """

    drt = []
    midrt = []

    if is_center:
        # order = 0
        drt.append(2*rt[0])
        midrt.append(0.0)

        # order > 0
        for order in range(1, orders):
            drt.append(rt[order] - rt[order - 1])
            midrt.append((rt[order] + rt[order - 1]) / 2)

    else:
        global_shift = rt[orders] / 2
        for order in range(1, orders):
            drt.append(rt[order] - rt[order - 1])
            midrt.append((rt[order] + rt[order - 1]) / 2 - global_shift)

    return np.array(drt, dtype=np.float32), np.array(midrt, dtype=np.float32)

"""
SIMULATION
"""
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

def get_Ibase_batch_simulation(
        monitor,
        sources_batch: list,
        cell,
        frequents_array: list = conf.inc_frequents_array,
        max_run_time: int = conf.maximum_run_time,
        symmetries=None,
        filename: str = None,
        verbose = False,
):
    """
    Wrapper: run broadband simulation with different source angles
    and return IPlane_base results.

    Parameters:
        monitor: single NearRegion monitor
        sources_batch : list of source lists
        cell : simulation cell size
        frequents_array : list of frequencies to monitor
        max_run_time : maximum simulation time
        symmetries : optional symmetries for simulation
        filename : optional filename to save/load results
    Returns:
        IPlane_base_all : simulation results for a set of angels
    """

    if filename is None:
        filename = f"basePlane_D{conf.aperture}_f{conf.f}_diaph{conf.diaphragm}_res{conf.coef_resolution}_nfreq{conf.inc_nfreq}.npy"
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
        if verbose:
            print(f"Running batch: {idx}/{n_sources}...")
        IBase_single_angle = get_Ibase_simulation(
            monitor=monitor,
            sources=sources,
            cell=cell,
            frequents_array=frequents_array,
            max_run_time=max_run_time,
            symmetries=symmetries,
        )
        results.append(IBase_single_angle)

    # save
    np.save(filename, results)
    print(f"Simulation result saved to: {filename}")
    return np.array(results)

def get_IAiry_simulation(
        monitor,
        cell,
        resolution,
        frequents_array: list = conf.inc_frequents_array,
        max_run_time: int = conf.maximum_run_time,
        symmetries=None,
        epsilon=None,
        filename_focus_airy: str = None,
        filename_base_airy: str = None,
        get_base: bool = False,
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
    IAiry_focus = None
    IAiry_base = None

    if filename_focus_airy is None:
        filename_focus_airy = f"focusAiry_D{conf.aperture}_f{conf.f}_diaph{conf.diaphragm}_res{conf.coef_resolution}_nfreq{conf.inc_nfreq}.npy"
        filename_base_airy = f"baseAiry_D{conf.aperture}_f{conf.f}_diaph{conf.diaphragm}_res{conf.coef_resolution}_nfreq{conf.inc_nfreq}.npy"
    else:
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

    base_results = []
    results = []
    sim = None

    for freq in frequents_array:
        if get_base:
        # if len(base_results) == 0:

            # plane wave
            src = my_sources.EigenmodeSource(
                frequency=freq,
                diaphragm=conf.diaphragm,
                center=mp.Vector3(0,0,0),
                df=conf.inc_df,
                resolution=resolution,
                angle_in_rad=0,
                epsilon=epsilon,
                focus_point=None,
            ).get_sources()

            sim = mp.Simulation(
                cell_size=cell,
                boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.X)],
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
                f=conf.f
            )
            base_results.append(IAiry_base_single_wave)

        # focused wave
        src = my_sources.EigenmodeSource(
            frequency=freq,
            diaphragm=conf.diaphragm,
            center=mp.Vector3(0, 0, 0),
            df=conf.inc_df,
            resolution=resolution,
            angle_in_rad=0,
            epsilon=epsilon,
            focus_point=mp.Vector3(x=conf.f),
        ).get_sources()

        sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.X)],
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
            f=conf.f
        )
        results.append(IAiry_focus_single_wave)

    IAiry_focus = np.stack(results, axis=1) # Stack results to array shape (ff_res, nfreq)
    if get_base:
        IAiry_base = np.stack(base_results, axis=1)

    np.save(filename_focus_airy, IAiry_focus)
    print(f"Simulation focus result saved to: {filename_focus_airy}")
    np.save(filename_base_airy, IAiry_base)
    print(f"Simulation base result saved to: {filename_base_airy}")
    return IAiry_focus, IAiry_base, sim

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
        boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.X)],
        sources=sources,
        geometry=geometry,
        resolution=conf.min_resolution * conf.coef_resolution,
        symmetries=symmetries,
    )
    nf2_list = []

    for freq in frequents_array:
        nf2_list.append(
            sim.add_near2far(
                freq,
                conf.inc_df,
                1,
                monitor
            )
        )

    sim.run(until=max_run_time)
    IBase_allwave = get_I(
        sim,
        nf2_list,
        conf.ff_res,
        mp.Vector3(0, conf.spot_ff, 0),
        f=conf.f - conf.add_size_cell / 2
    )
    return IBase_allwave

def get_I(sim, near2far_region, ff_res, size, f):
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
            center=mp.Vector3(x=f, y=0),
            size=size
        )
        return abs(far_field["Ez"]) ** 2

    if isinstance(near2far_region, (list, tuple)):
        # Если передан список регионов, вычисляем для каждого и собираем
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
    :param f: фокальная плоскость
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

def plot_intensity(I_ff, out_angle_in_deg, wavelength_array, abscissa,
                   I_norm=None, isfwhm=True, save_path: str = None):
    """
    :param I_ff: intensity values (normalized from 0 to 1 or arbitrary scale)
    :param out_angle_in_deg: output refraction angle
    :param wavelength_array: array of wavelengths
    :param abscissa: X-axis coordinates
    :param I_norm: 1D array of intensity values for normalization reference
    :param isfwhm: print flag for FWHM
    :param save_path: optional path to save figure as PDF (if None, figure is not saved)
    """

    # --- Нормализация ---
    normalized = False
    if I_norm is not None:
        if I_norm.ndim != 1:
            raise ValueError("I_norm must be a 1D array.")
        norm_max = np.max(I_norm)
        if norm_max != 0:
            I_ff = I_ff / norm_max
            normalized = True
        else:
            print("⚠️ Warning: I_norm has zero maximum, normalization skipped.")

    # --- Настройки оформления ---
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
    })

    plt.style.use('default')
    plt.figure(dpi=100)

    # --- Построение ---
    if len(wavelength_array) == 1:
        plt.plot(abscissa, I_ff, label=f"Far field ({wavelength_array[0]} μm)")
    else:
        for i in range(I_ff.shape[1]):
            plt.plot(abscissa, I_ff[:, i], label=f"Far field (λ={wavelength_array[i]} μm)")

    # --- Подписи и оформление ---
    plt.xlabel("x coordinate [µm]")
    plt.ylabel(r"Normalized energy density, |E$_z$|$^2$")
    plt.title(f"Intensity of CC lens. Out Incident {np.around(out_angle_in_deg, decimals=1)}°")
    plt.legend(loc="upper right")
    plt.tight_layout()

    # --- Сохранение ---
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"📄 Figure saved to: {save_path}")

    plt.show()

    # --- FWHM ---
    if isfwhm:
        if normalized:
            # При нормализации: анализ всех длин волн вместе
            fwhm_values = get_fwhm(abscissa, I_ff)
            for i, fwhm in enumerate(fwhm_values):
                print(f"λ={wavelength_array[i]}: FWHM = {fwhm:.2f} µm")
        else:
            # Без нормализации считаем для каждой длины волны отдельно
            if len(wavelength_array) == 1:
                fwhm = get_fwhm(abscissa, I_ff)
                print(f"FWHM = {fwhm:.2f} µm")
            else:
                for i in range(I_ff.shape[1]):
                    fwhm = get_fwhm(abscissa, I_ff[:, i])
                    print(f"λ={wavelength_array[i]}: FWHM = {fwhm:.2f} µm")

def get_fwhm(abscissa, intensity):
    """
    Calculate FWHM (Full Width at Half Maximum) for 1D or 2D intensity profiles.

    :param abscissa: x-coordinates
    :param intensity: 1D or 2D array of intensities
    :return: float (for 1D) or list[float] (for 2D)
    """
    # --- Универсальная функция для одного профиля ---
    def _fwhm_single(x, I):
        center = np.argmax(I)
        n = len(I)

        def find_width(level):
            left_idx, right_idx = None, None
            for i in range(center, -1, -1):
                if I[i] <= level:
                    left_idx = i
                    break
            for i in range(center, n):
                if I[i] <= level:
                    right_idx = i
                    break
            if left_idx is None or right_idx is None:
                return np.nan
            return float(x[right_idx] - x[left_idx])

        I_norm = I / np.max(I) if np.max(I) != 0 else I
        return find_width(0.5)

    # --- Если одномерный профиль ---
    if intensity.ndim == 1:
        return _fwhm_single(abscissa, intensity)

    # --- Если многомерный: нормализуем все профили на глобальный максимум ---
    global_max = np.max(intensity)
    if global_max == 0:
        raise ValueError("Intensity array has zero maximum, cannot compute FWHM.")

    I_norm = intensity / global_max

    fwhm_list = []
    for i in range(I_norm.shape[1]):
        fwhm_i = _fwhm_single(abscissa, I_norm[:, i])
        fwhm_list.append(fwhm_i)

    return fwhm_list

# def get_focusing_efficiency(
#         airy_focus: np.ndarray,
#         airy_base: np.ndarray,
#         abscissa: np.ndarray,
#         test_focus: Optional[np.ndarray] = None,
#         test_base: Optional[np.ndarray] = None
# ) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
#     """
#     Calculate focusing efficiency metrics.
#
#     Parameters
#     ----------
#     airy_focus : np.ndarray
#         Focused Airy pattern profile(s).
#     airy_base : np.ndarray
#         Base/background level for Airy pattern.
#     abscissa : np.ndarray
#         Spatial coordinates along the profile.
#     test_focus : Optional[np.ndarray], optional
#         Test profile(s) to analyze.
#     test_base : Optional[np.ndarray], optional
#         Base/background level for test profile.
#
#     Returns
#     -------
#     tuple
#         (arr_STREHL, arr_EER_airy, arr_EER_test, arr_EER_rel)
#         If test data is not provided, returns (None, arr_EER_airy, None, None)
#     """
#     # Validate Airy input shapes
#     if airy_focus.shape != airy_base.shape:
#         raise ValueError("airy_focus and airy_base must have the same shape")
#
#     if airy_focus.shape[0] != abscissa.size:
#         raise ValueError("First dimension of focus arrays must match abscissa length")
#
#     # Calculate metrics that require only Airy data
#     dims, waves = airy_focus.shape
#     dx = np.mean(np.diff(abscissa))
#     center = 0.5 * (abscissa[0] + abscissa[-1])
#     airy_diam = np.atleast_1d(conf.airy_diameter)
#
#     arr_EER_airy = np.zeros(waves)
#
#     for wave in range(waves):
#         mask = (abscissa >= center - airy_diam[wave] / 2) & (abscissa <= center + airy_diam[wave] / 2)
#
#         power_in_airy = np.sum(airy_focus[mask, wave]) * dx
#         power_total_airy = np.sum(airy_base[:, wave]) * dx
#
#         arr_EER_airy[wave] = power_in_airy / power_total_airy if power_total_airy > 0 else 0.0
#
#     if test_focus is None or test_base is None:
#         return None, arr_EER_airy, None, None
#
#     if test_focus.shape != test_base.shape:
#         raise ValueError("test_focus and test_base must have the same shape")
#
#     if airy_focus.shape != test_focus.shape:
#         raise ValueError("Airy and test arrays must have the same shape")
#
#     # Вычисляем остальные метрики
#     airy_flux_norm = airy_focus / airy_base
#     test_flux_norm = test_focus / test_base
#     arr_STREHL = np.max(test_flux_norm, axis=0) / np.max(airy_flux_norm, axis=0)
#
#     arr_EER_test = np.zeros(waves)
#     arr_EER_rel = np.zeros(waves)
#
#     for wave in range(waves):
#         mask = (abscissa >= center - airy_diam[wave] / 2) & (abscissa <= center + airy_diam[wave] / 2)
#
#         power_in_test = np.sum(test_focus[mask, wave]) * dx
#         power_total_test = np.sum(test_base[:, wave]) * dx
#
#         arr_EER_test[wave] = power_in_test / power_total_test if power_total_test > 0 else 0.0
#         arr_EER_rel[wave] = arr_EER_test[wave] / arr_EER_airy[wave] if arr_EER_airy[wave] > 0 else 0.0
#
#     return arr_STREHL, arr_EER_airy, arr_EER_test, arr_EER_rel

def get_focusing_efficiency(
        airy_focus: np.ndarray,
        abscissa: np.ndarray,
        test_focus: np.ndarray,
        airy_base: Optional[np.ndarray] = None,
        test_base: Optional[np.ndarray] = None,
        *,
        call_id: Optional[int] = None,
        results_filename: Optional[str] = None
) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        float, float, float, float
]:
    """
    Calculate focusing efficiency metrics.

    Returns
    -------
    arr_STREHL : np.ndarray
    arr_EER_airy : np.ndarray
    arr_EER_test : np.ndarray
    arr_EER_rel : np.ndarray
    mean_STREHL : float
    std_STREHL : float
    mean_EER_rel : float
    std_EER_rel : float
    """

    # --- Validate shapes ---
    if airy_focus.shape[0] != abscissa.size:
        raise ValueError("First dimension of airy_focus must match abscissa length")

    if airy_base is not None and airy_base.shape != airy_focus.shape:
        raise ValueError("airy_base must have the same shape as airy_focus")

    if test_focus.shape[0] != abscissa.size:
        raise ValueError("First dimension of test_focus must match abscissa length")

    if test_base is not None and test_base.shape != test_focus.shape:
        raise ValueError("test_base must have the same shape as test_focus")

    if airy_focus.shape != test_focus.shape:
        raise ValueError("Airy and test arrays must have the same shape")

    dims, waves = airy_focus.shape
    dx = float(np.mean(np.diff(abscissa)))
    center = 0.5 * (abscissa[0] + abscissa[-1])
    airy_diam = np.atleast_1d(conf.airy_diameter)

    # === EER for Airy ===
    arr_EER_airy = np.zeros(waves)

    for wave in range(waves):
        mask = (
            (abscissa >= center - airy_diam[wave] / 2) &
            (abscissa <= center + airy_diam[wave] / 2)
        )

        power_in_airy = np.sum(airy_focus[mask, wave]) * dx

        if airy_base is not None:
            power_total_airy = np.sum(airy_base[:, wave]) * dx
        else:
            power_total_airy = np.sum(airy_focus[:, wave]) * dx

        arr_EER_airy[wave] = (
            power_in_airy / power_total_airy if power_total_airy > 0 else 0.0
        )

    # === Normalization for Strehl ===
    airy_norm = np.zeros_like(airy_focus)
    test_norm = np.zeros_like(test_focus)

    for wave in range(waves):
        if airy_base is not None:
            base = airy_base[:, wave]
            np.divide(
                airy_focus[:, wave],
                base,
                out=airy_norm[:, wave],
                where=base != 0
            )
        else:
            energy = np.sum(airy_focus[:, wave]) * dx
            airy_norm[:, wave] = airy_focus[:, wave] / energy if energy > 0 else 0.0

        if test_base is not None:
            base = test_base[:, wave]
            np.divide(
                test_focus[:, wave],
                base,
                out=test_norm[:, wave],
                where=base != 0
            )
        else:
            energy = np.sum(test_focus[:, wave]) * dx
            test_norm[:, wave] = test_focus[:, wave] / energy if energy > 0 else 0.0

    # === Strehl ratio ===
    airy_max = np.max(airy_norm, axis=0)
    test_max = np.max(test_norm, axis=0)
    arr_STREHL = np.where(airy_max > 0, test_max / airy_max, 0.0)

    # === EER for test and relative EER ===
    arr_EER_test = np.zeros(waves)
    arr_EER_rel = np.zeros(waves)

    for wave in range(waves):
        mask = (
            (abscissa >= center - airy_diam[wave] / 2) &
            (abscissa <= center + airy_diam[wave] / 2)
        )

        power_in_test = np.sum(test_focus[mask, wave]) * dx

        if test_base is not None:
            power_total_test = np.sum(test_base[:, wave]) * dx
        else:
            power_total_test = np.sum(test_focus[:, wave]) * dx

        arr_EER_test[wave] = (
            power_in_test / power_total_test if power_total_test > 0 else 0.0
        )
        arr_EER_rel[wave] = (
            arr_EER_test[wave] / arr_EER_airy[wave] if arr_EER_airy[wave] > 0 else 0.0
        )

    # === Aggregated metrics ===
    mean_STREHL = float(np.mean(arr_STREHL))
    std_STREHL = float(np.std(arr_STREHL))
    mean_EER_rel = float(np.mean(arr_EER_rel))
    std_EER_rel = float(np.std(arr_EER_rel))

    # === Save results (append to file) ===
    if results_filename is not None:
        write_header = not os.path.exists(results_filename)

        with open(results_filename, "a", encoding="utf-8") as f:
            if write_header:
                f.write(
                    "call_id,angle,wave,STREHL,EER_airy,EER_test,EER_rel,"
                    "mean_STREHL,std_STREHL,mean_EER_rel,std_EER_rel\n"
                )

            for wave in range(waves):
                f.write(
                    f"{call_id},{conf.angle_list[call_id]},{conf.inc_wavelength_array[wave]},"
                    f"{arr_STREHL[wave]},"
                    f"{arr_EER_airy[wave]},"
                    f"{arr_EER_test[wave]},"
                    f"{arr_EER_rel[wave]},"
                    f"{mean_STREHL},"
                    f"{std_STREHL},"
                    f"{mean_EER_rel},"
                    f"{std_EER_rel}\n"
                )

    return (
        arr_STREHL,
        arr_EER_airy,
        arr_EER_test,
        arr_EER_rel,
        mean_STREHL,
        std_STREHL,
        mean_EER_rel,
        std_EER_rel,
    )

def plot_psf(
    nfrec: int,
    spot_ff: Union[float, Sequence[float]],
    I: np.ndarray,
    abscissa,
    save_path: str = None,
    save_format: str = "pdf",
    colorbar_to_1: bool = True,
) -> None:
    """
    Builds and displays a PSF (Point Spread Function) based on a one-dimensional or two-dimensional
    intensity distribution. Supports saving to PDF or JPEG.

    :param nfrec : int
        The number of subgraphs (wavelength discreteness).
    :param spot_ff : float or array_like
        Spatial area (for example, an aperture) on which the intensities are set.
    :param I : ndarray
        Array of intensities:
        - if the shape is (M,), then one PSF will be constructed;
        - if the shape is (M, N), then N PSFs will be constructed.
    :param abscissa : array_like
        An array of coordinates along the X-axis.
    :param save_path : str, optional
        The path to save the file (for example, "psf_plot.pdf" or "psf_plot.jpeg").
        If None, the graph is simply displayed on the screen.
    :param save_format : str, optional
        Save format: "pdf" or "jpeg". The default is "pdf".
    :param normalize : bool, optional
        Whether to normalize intensity maps to their maximum values (default: True).
    """

    # --- Настройки оформления ---
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
    })

    # Проверка формы входных данных
    I = np.asarray(I)
    if I.ndim == 1:
        I = I[:, np.newaxis]
    elif I.ndim != 2:
        raise ValueError(f"Ожидается 1D или 2D массив, получили ndim={I.ndim}")

    # Координаты
    x_plot = abscissa
    x_calculate = get_abscissa(spot_ff, I[:, 0], 0)

    # Сетка
    X, Y = np.meshgrid(x_calculate, x_calculate)
    R = np.hypot(X, Y)

    # Подготовка фигуры
    fig, axes = plt.subplots(1, nfrec, figsize=(6 * nfrec, 6), squeeze=False, dpi=300)
    axes = axes.ravel()
    wavelength_array = conf.inc_wavelength_array

    for idx in range(nfrec):
        if idx >= I.shape[1]:
            intensity = np.zeros_like(x_calculate)
        else:
            intensity = I[:, idx]

        # Интерполяция
        interp_func = interp1d(x_calculate, intensity, bounds_error=False, fill_value=0.0)
        I2d = interp_func(R)

        # Нормировка по флагу
        if colorbar_to_1:
            I2d -= I2d.min()
            if I2d.max() > 0:
                I2d /= I2d.max()

        # Отображение
        im = axes[idx].imshow(
            I2d,
            extent=(x_plot[0], x_plot[-1], x_calculate[0], x_calculate[-1]),
            origin="lower",
            cmap="viridis",
            aspect="equal",
        )

        title = (
            f"PSF for {conf.inc_wavelength:.1f} μm"
            if len(wavelength_array) == 1
            else f"PSF for {wavelength_array[idx]:.1f} μm"
        )
        axes[idx].set(
            title=title,
            xlabel="Spot [μm]",
            ylabel="Spot [μm]",
        )
        cbar_label = "Normalized intensity" if colorbar_to_1 else "Intensity"
        fig.colorbar(im, ax=axes[idx], label=cbar_label)

    # --- Заголовок и разметка ---
    plt.tight_layout()
    fig.suptitle(f"Resolution: {len(x_plot)} pixels", fontsize=16, color="blue", y=1.02)

    # --- Сохранение / вывод ---
    if save_path:
        save_format = save_format.lower()
        if save_format not in ["pdf", "jpeg"]:
            raise ValueError("save_format must be 'pdf' or 'jpeg'")

        fig.savefig(save_path, format=save_format, bbox_inches="tight")
        print(f"PSF plot saved to: {save_path}")

    else:
        plt.show()


def plot_mtf(
        spot_ff: Union[float, Sequence[float]],
        I_ff: np.ndarray,
        in_angle_in_deg: float,
        out_angle_in_deg: float,
        max_view: float = 50,
        smooth: bool = False,
        dif_lim: bool = False,
        I_mtf_theory: np.ndarray = None,
        save_path: str = None,
        save_format: str = "pdf"
) -> None:
    """
    Builds an MTF curve based on the intensity distribution in the focal plane.
    It can save the result to PDF or JPEG.

    Parameters
    ----------
spot_ff : float or array_like
Spatial area (for example, an aperture).
    I_ff : ndarray
        An array of PSF intensities (1D or 2D).
    in_angle_in_deg : float
        The angle at the entrance.
    out_angle_in_deg : float
        The angle at the exit.
    max_view : float
        The maximum frequency on the graph (lp/mm).
    smooth : bool
        Smoothing of calculated MTFs.
    dif_lim : bool
        Include values for the ideal case.
    I_mtf_theory : ndarray
        The pre-calculated values of the ideal MTF.
    save_path : str
        The path to the file to save (for example, "mtf_plot.pdf" or "mtf_plot.jpeg ").
        If None, the graph is simply displayed on the screen.
    save_format : str
        Save format: "pdf" or "jpeg".
    """

    # --- Design Settings ---
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
    })

    I_ff = np.asarray(I_ff)
    if I_ff.ndim == 1:
        I_ff = I_ff[:, np.newaxis]

    # Получение пространственных координат
    x = get_abscissa(spot_ff, I_ff[:, 0], angle=0)
    dx = np.abs(x[1] - x[0])
    dx_mm = dx * 1e-3
    N = len(x)

    freq = np.fft.fftfreq(N, d=dx_mm)[:N // 2 + 1]

    wavelength_array = conf.inc_wavelength_array
    theta_rad = np.radians(in_angle_in_deg)

    fig, ax = plt.subplots(dpi=300)

    # --- Основной расчёт и построение ---
    for i in range(I_ff.shape[1]):
        psf = I_ff[:, i]
        psf /= np.sum(psf)

        mtf = np.abs(fft(psf))[:N // 2 + 1]
        mtf_normalized = mtf / mtf.max()

        if smooth:
            window_length = 7
            polyorder = 2
            mtf_normalized = savgol_filter(mtf_normalized, window_length, polyorder)

        label = f"optimized ({wavelength_array[i]:.1f} μm)"
        ax.plot(freq, mtf_normalized, label=label)

    # --- Дифракционный предел ---
    if dif_lim:
        if I_mtf_theory is not None:
            for i in range(I_mtf_theory.shape[1]):
                psf = I_mtf_theory[:, i]
                psf /= np.sum(psf)
                mtf = np.abs(fft(psf))[:N // 2 + 1]
                mtf_normalized = mtf / mtf.max()

                if smooth:
                    window_length = 7
                    polyorder = 2
                    mtf_normalized = savgol_filter(mtf_normalized, window_length, polyorder)

                ax.plot(freq, mtf_normalized, linestyle='--',
                        label=f"limit_sim ({wavelength_array[i]:.1f} μm)")
        else:
            for lam in wavelength_array:
                lam_mm = lam * 1e-3
                fc = 1 / lam_mm
                fc_eff = fc * np.cos(theta_rad)
                f_ratio = np.linspace(0, fc_eff, 200)
                mtf_th = (2 / np.pi) * (
                    np.arccos(f_ratio / fc_eff)
                    - (f_ratio / fc_eff) * np.sqrt(1 - (f_ratio / fc_eff) ** 2)
                )
                ax.plot(f_ratio, mtf_th, linestyle='--', label=f'dif_lim {lam:.2f} µm')

    # --- Adj plots ---
    ax.set_xlim(0, max_view)
    ax.set_xlabel("Spatial Frequency (lp/mm)")
    ax.set_ylabel("Contrast (MTF)")
    ax.set_title(f"Modulation Transfer Function for {np.around(out_angle_in_deg, 1)}°")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(False)
    fig.tight_layout()

    # --- Save ---
    if save_path:
        save_format = save_format.lower()
        if save_format not in ["pdf", "jpeg"]:
            raise ValueError("save_format must be either 'pdf' or 'jpeg'")
        fig.savefig(save_path, format=save_format, bbox_inches="tight")
        print(f"MTF plot saved to: {save_path}")

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

def polygon_area(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )


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
        precision: float = 5e-9,
        MIN_POLY_AREA: float = 0.05  # μm²
) -> None:
    """
    Генерирует многослойный GDSII файл.

    Этап 1:
        Проверка площади одного сектора для каждого порядка m.
        Формируется список допустимых порядков.

    Этап 2:
        Геометрия создаётся только для допустимых порядков.
    """

    a_params = np.asarray(a_params, dtype=float)
    h_params = np.asarray(h_params, dtype=float)
    rt = np.asarray(rt, dtype=float)

    # ============================================================
    # ЭТАП 1 — ПРЕДВАРИТЕЛЬНАЯ ПРОВЕРКА ПЛОЩАДЕЙ
    # ============================================================

    valid_orders = np.ones(orders, dtype=bool)

    print("=== Area pre-check per order ===")

    for m in range(orders):
        a = max(a_params[m], 1e-3)

        rt0 = 0.0 if m == 0 else rt[m - 1] + 0.0025
        rt1 = rt[m] - 0.0025
        # rt0 = 0.0 if m == 0 else rt[m - 1]
        # rt1 = rt[m]
        x0 = 0.0 if m == 0 else 0.5 * (rt[m] + rt0)
        r = x0

        if r < 1e-6:
            valid_orders[m] = False
            print(f"[SKIP] order={m}, reason=r too small")
            continue

        L = np.clip(1.5 / a, 2.0, 10.0)
        x_left = max(rt0, x0 - L * a)
        x_right = min(rt1, x0 + L * a)

        if x_right <= x_left:
            valid_orders[m] = False
            print(f"[SKIP] order={m}, reason=empty x-range")
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

        xx_local = xx - x0

        test_coords = np.vstack([
            xx_local + r,
            yy
        ]).T

        area = polygon_area(test_coords)

        if area < MIN_POLY_AREA:
            valid_orders[m] = False
            print(
                f"[SKIP] order={m}, "
                f"area={area:.4e} μm² < MIN_POLY_AREA={MIN_POLY_AREA:.4e} μm²"
            )

    # ---------- СВОДКА ----------
    print("=== Area check completed ===\n")
    total_orders = orders
    rejected_orders = np.count_nonzero(~valid_orders)
    accepted_orders = np.count_nonzero(valid_orders)

    print("\n=== Area check summary ===")
    print(f"Total orders checked : {total_orders}")
    print(f"Rejected orders      : {rejected_orders}")
    print(f"Accepted orders      : {accepted_orders}")
    print("==========================\n")

    # ============================================================
    # ЭТАП 2 — ГЕНЕРАЦИЯ GDS
    # ============================================================

    lib = gdspy.GdsLibrary(unit=unit, precision=precision)
    cell = lib.new_cell(cell_name)

    layer_height = 0.1  # μm

    for layer in range(n_layers):
        current_height = layer * layer_height

        for m in range(orders):
            if not valid_orders[m]:
                continue

            a = max(a_params[m], 1e-3)
            h = h_params[m]

            if h < current_height - 1e-12:
                continue

            rt0 = 0.0 if m == 0 else rt[m - 1] + 0.0025
            rt1 = rt[m] - 0.0025
            # rt0 = 0.0 if m == 0 else rt[m - 1]
            # rt1 = rt[m]
            x0 = 0.0 if m == 0 else 0.5 * (rt[m] + rt0)
            r = x0

            L = np.clip(1.5 / a, 2.0, 10.0)
            x_left = max(rt0, x0 - L * a)
            x_right = min(rt1, x0 + L * a)

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

            xx_local = xx - x0

            w_rad_cell = w / r
            w_rad_gap = w / r
            T_rad = w_rad_cell + w_rad_gap
            N = max(1, int(np.floor(2 * np.pi / T_rad)))

            for n in range(N):
                theta0 = n * T_rad
                cos0 = np.cos(theta0)
                sin0 = np.sin(theta0)

                coords = np.vstack([
                    xx_local * cos0 - yy * sin0 + r * cos0,
                    xx_local * sin0 + yy * cos0 + r * sin0
                ]).T

                cell.add(gdspy.Polygon(coords, layer=layer))

    lib.write_gds(filename)
    print(f"GDS saved: {filename}")

# works
def generate_lens_stl_cc_linear(
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
    Generates an STL model of the lens on the substrate.
    The cells are placed linearly (±N copies on the Y axis) and mirrored on the X axis.
    All dimensions are in microns.
    """
    a_params = np.asarray(a_params, dtype=float)
    h_params = np.asarray(h_params, dtype=float)
    rt = np.asarray(rt, dtype=float)
    substrate_thickness = substrate_thickness

    meshes = []

    # ---- substrate ----
    W = 20
    L = rt[-1] * 2
    H = substrate_thickness
    substrate = trimesh.creation.box(extents=(L, W, H))
    substrate.apply_translation((0, 0, H / 2))
    meshes.append(substrate)

    # ---- lens segments ----
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

        # ---- Generating copies along Y and mirroring along X ----
        y_offsets = np.arange(-N, N+1) * w * 2
        x_signs = [1, -1]  # mirror symmetry in X

        for x_sign in x_signs:
            for y_off in y_offsets:
                coords = np.vstack([xx * x_sign, yy + y_off]).T
                poly = Polygon(coords)

                # Clearing self-intersections and orientation
                if not poly.is_valid:
                    try:
                        poly = poly.buffer(0)
                        poly = orient(poly)
                        if not poly.is_valid:
                            from shapely.validation import explain_validity
                            print(f"После buffer(0) полигон всё ещё невалиден для сегмента {m}, смещение {y_off}")
                            print(f"Причина: {explain_validity(poly)}")
                            continue
                    except Exception as e:
                        print(f"Ошибка при очистке полигона для сегмента {m}, смещение {y_off}: {e}")
                        continue

                if poly.area < 1e-6:
                    print(f"Малая площадь полигона ({poly.area:.6f}) для сегмента {m}, смещение {y_off}")

                # Экструзия на высоту
                try:
                    tri = trimesh.creation.extrude_polygon(poly, h, engine="triangle")
                except Exception:
                    tri = trimesh.creation.extrude_polygon(poly, h)

                tri.apply_translation((0, 0, substrate_thickness))
                meshes.append(tri)

    # ---- Final mesh ----
    if meshes:
        final_mesh = trimesh.util.concatenate(meshes)
        final_mesh.export(filename)
        print(f"STL saved: {filename}")
    else:
        print("Нет геометрии для экспорта")


def generate_lens_gds_cc_linear(
        a_params,
        rt,
        orders: int,
        w: float = 0.1,
        Q: int = 100,
        W: float = None,
        filename: str = "lens_layout_linear.gds"
) -> None:
    """
    Генерирует GDS-модель линзы.
    Ячейки размещаются линейно (±N копий по оси Y) и зеркально по оси X.
    Все размерности в микронах.

    Параметры:
    ----------
    a_params : array-like
        Оптимизационные параметры формы (по числу сегментов).
    rt : array-like
        Радиусы границ сегментов (в микронах).
    orders : int
        Количество сегментов.
    w : float
        Толщина одной полосы (ширина мета-ячейки) в микронах.
    Q : int
        Количество точек для аппроксимации профиля.
    W : float | None
        Общая половинная ширина структуры вдоль оси Y.
        Если не задана, используется максимальный радиус из rt.
    filename : str
        Имя файла GDS для сохранения.
    """
    a_params = np.asarray(a_params, dtype=float)
    rt = np.asarray(rt, dtype=float)

    if W is None:
        W = float(rt[-1])
    N = int(np.floor(W / (2 * w)))  # число копий по оси Y

    cell = gdspy.Cell("LENS")

    for m in range(orders):
        a = max(a_params[m], 1e-3)

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

        # центрирование профиля
        ip = np.argmin(y_profile)
        y_shifted = y_profile - y_profile[ip]

        ymax = 0.5 * (x.max() - x.min())
        y_top = np.clip(y_shifted, -ymax, ymax)
        y_bot = np.clip(y_shifted - w, -ymax, ymax)

        xx = np.concatenate([x, x[::-1]])
        yy = np.concatenate([y_top, y_bot[::-1]])

        # ---- генерация копий вдоль Y и зеркальные по X ----
        y_offsets = np.arange(-N, N + 1) * w * 2
        x_signs = [1, -1]

        for x_sign in x_signs:
            for y_off in y_offsets:
                coords = np.vstack([xx * x_sign, yy + y_off]).T
                poly = Polygon(coords)

                if not poly.is_valid:
                    poly = poly.buffer(0)
                    poly = orient(poly)
                    if not poly.is_valid:
                        continue

                if poly.area < 1e-6:
                    continue

                points = np.array(poly.exterior.coords)
                cell.add(gdspy.Polygon(points, layer=1))

    gdspy.write_gds(filename)
    print(f"GDS saved: {filename}")

def generate_lens_stl_grid(
        a_params,      # (orders,)
        h_params,      # (orders,)
        orders: int,
        rt_np,
        stack: int,
        substrate_thickness: float,  # H
        LIMIT_SHAPE: float = 0.0,
        filename: str = "diffraction_grating.stl",
        Q: int = 200,
        w: float = 0.1,
) -> trimesh.Trimesh:
    """
    Forms a diffraction grating in the STL.
    The intervals are set by rectangular blocks, the height is taken from h_params.
    """

    # --- входные данные в NumPy ---
    a_np = np.asarray(a_params, dtype=float)
    h_np = np.asarray(h_params, dtype=float)

    # служебные массивы из внешней функции (предполагаем, что работают с numpy)
    list_drt, list_crt = get_drt_midrt(rt_np, orders)
    y0 = -0.0001 + 2 * stack * w

    rects = []  # список прямоугольников (x_center_local, length_x, height, x0)

    # --- цикл по порядкам ---
    for m in range(orders):
        a = np.clip(a_np[m], 1e-3, None)

        if m == 0:
            rt0 = 0.0
        else:
            rt0 = rt_np[m - 1] + 0.01
        x0 = 0.0 if m == 0 else (rt_np[m] + rt0) / 2.0

        x = np.linspace(rt0, rt_np[m] - 0.01, Q)
        z = np.clip((x - x0) / a, -20.0, 20.0)
        y_profile = a * np.cosh(z)

        pure_x_list = []
        for n in range(stack, -1, -1):
            ip = int(np.argmin(y_profile))
            y_shifted = y_profile - y_profile[ip]
            ymax = 0.5 * (x.max() - x.min())
            y_top = np.clip(y_shifted, -ymax, ymax)
            y_bot = np.clip(y_shifted - w, -ymax, ymax)

            xx = np.concatenate([x, x[::-1]])
            yy = np.concatenate([y_top, y_bot[::-1]]) + 2.0 * n * w

            x_int = cut_y(xx, yy, y0)
            pure_x_list.append(x_int)

        # нормировка точек
        filtered = []
        for arr in pure_x_list:
            if arr is None or arr.size == 0:
                continue
            L = arr.size
            if L == 1:
                arr = np.array([-arr[0], arr[0]], dtype=float)
            elif L == 2 and m == 0:
                v0, v1 = float(arr[0]), float(arr[1])
                arr = np.array([-v1, -v0, v0, v1], dtype=float)
            elif L in (2, 4):
                arr = arr - list_crt[m]
            else:
                continue
            filtered.append(arr)

        # превращаем интервалы в прямоугольники
        for arr in filtered:
            if arr.size == 2:
                intervals = [(float(arr[0]), float(arr[1]))]
            elif arr.size == 3:
                intervals = [(float(arr[0]), float(arr[1])),
                             (-float(arr[1]), -float(arr[0]))]
            else:
                intervals = [(float(arr[0]), float(arr[1])),
                             (float(arr[2]), float(arr[3]))]

            for (l, r) in intervals:
                length_x = float(r) - float(l)
                if length_x <= LIMIT_SHAPE:
                    continue
                x_center_local = (float(r) + float(l)) / 2.0
                rects.append((x_center_local, length_x, float(h_np[m]), x0))

    # --- строим геометрию ---
    meshes = []

    W = 20.0
    L = float(rt_np[-1]) * 2.0
    H = float(substrate_thickness)

    substrate = trimesh.creation.box(extents=(L, W, H))
    substrate.apply_translation((0.0, 0.0, H / 2.0))
    meshes.append(substrate)

    for x_center_local, length_x, height, x0 in rects:
        x_center_global = x_center_local + x0
        box = trimesh.creation.box(extents=(length_x, W, height))
        box.apply_translation((x_center_global, 0.0, H + height / 2.0))
        meshes.append(box)

        # зеркальная копия
        mirror_box = box.copy()
        mirror_box.apply_transform(np.diag([-1, 1, 1, 1]))
        meshes.append(mirror_box)

    if meshes:
        combined = trimesh.util.concatenate(meshes)
        combined.export(filename)
        print(f"STL сохранён: {filename}")
    else:
        print("Нет геометрии для экспорта в STL")

def generate_lens_gds_grid(
    a_params,             # (orders,)
    orders: int,
    rt_np,
    stack: int,
    LIMIT_SHAPE: float = 0.0,
    filename: str = "diffraction_grating.gds",
    Q: int = 200,
    w: float = 0.1,
    W: float = None,
) -> None:
    """
    Формирует дифракционную решётку в GDS-формате.
    Интервалы задают прямоугольные блоки без учёта высоты.
    Подложка не используется. Все размерности — в микронах.

    Параметры:
    ----------
    a_params : array-like
        Параметры профиля (по числу сегментов).
    orders : int
        Количество сегментов.
    rt_np : array-like
        Радиусы границ сегментов.
    stack : int
        Количество уровней укладки (по оси Y, в единицах шагов w).
    LIMIT_SHAPE : float
        Минимальная допустимая длина прямоугольника по X.
    filename : str
        Имя выходного файла GDS.
    Q : int
        Количество точек для аппроксимации профиля.
    w : float
        Толщина одной линии в микронах.
    W : float | None
        Общая половинная ширина структуры вдоль оси Y. Если None — берётся из rt_np[-1].
    """
    # --- подготовка данных ---
    a_np = np.asarray(a_params, dtype=float)
    rt_np = np.asarray(rt_np, dtype=float)

    if W is None:
        W = float(rt_np[-1])

    # служебные массивы — их нужно получить из внешних функций
    list_drt, list_crt = get_drt_midrt(rt_np, orders)

    y0 = -0.0001 + 2 * stack * w
    rects = []  # (x_center_local, length_x, x0)

    # --- цикл по порядкам ---
    for m in range(orders):
        a = np.clip(a_np[m], 1e-3, None)
        rt0 = 0.0 if m == 0 else rt_np[m - 1] + 0.01
        x0 = 0.0 if m == 0 else (rt_np[m] + rt0) / 2.0

        x = np.linspace(rt0, rt_np[m] - 0.01, Q)
        z = np.clip((x - x0) / a, -20.0, 20.0)
        y_profile = a * np.cosh(z)

        pure_x_list = []
        for n in range(stack, -1, -1):
            ip = int(np.argmin(y_profile))
            y_shifted = y_profile - y_profile[ip]
            ymax = 0.5 * (x.max() - x.min())
            y_top = np.clip(y_shifted, -ymax, ymax)
            y_bot = np.clip(y_shifted - w, -ymax, ymax)
            xx = np.concatenate([x, x[::-1]])
            yy = np.concatenate([y_top, y_bot[::-1]]) + 2.0 * n * w

            x_int = cut_y(xx, yy, y0)
            pure_x_list.append(x_int)

        # фильтрация и формирование интервалов
        filtered = []
        for arr in pure_x_list:
            if arr is None or arr.size == 0:
                continue
            L = arr.size
            if L == 1:
                arr = np.array([-arr[0], arr[0]], dtype=float)
            elif L == 2 and m == 0:
                v0, v1 = float(arr[0]), float(arr[1])
                arr = np.array([-v1, -v0, v0, v1], dtype=float)
            elif L in (2, 4):
                arr = arr - list_crt[m]
            else:
                continue
            filtered.append(arr)

        # превращаем интервалы в прямоугольники
        for arr in filtered:
            if arr.size == 2:
                intervals = [(float(arr[0]), float(arr[1]))]
            elif arr.size == 3:
                intervals = [(float(arr[0]), float(arr[1])),
                             (-float(arr[1]), -float(arr[0]))]
            else:
                intervals = [(float(arr[0]), float(arr[1])),
                             (float(arr[2]), float(arr[3]))]

            for (l, r) in intervals:
                length_x = float(r) - float(l)
                if length_x <= LIMIT_SHAPE:
                    continue
                x_center_local = (float(r) + float(l)) / 2.0
                rects.append((x_center_local, length_x, x0))

    # --- построение геометрии GDS ---
    cell = gdspy.Cell("DIFF_GRATING")

    for x_center_local, length_x, x0 in rects:
        x_center_global = x_center_local + x0
        y0 = -W
        y1 = W
        x_left = x_center_global - length_x / 2.0
        x_right = x_center_global + length_x / 2.0

        # основной прямоугольник
        rect_coords = [
            (x_left, y0),
            (x_right, y0),
            (x_right, y1),
            (x_left, y1)
        ]
        cell.add(gdspy.Polygon(rect_coords, layer=1))

        # зеркальная копия по X
        rect_coords_mirror = [
            (-x_left, y0),
            (-x_right, y0),
            (-x_right, y1),
            (-x_left, y1)
        ]
        cell.add(gdspy.Polygon(rect_coords_mirror, layer=1))

    gdspy.write_gds(filename)
    print(f"GDS сохранён: {filename}")

def generate_lens_stl_cells(
        a_params,
        h_params,
        orders: int,
        rt_np,
        stack: int,
        substrate_thickness: float,
        LIMIT_SHAPE: float = 0.0,
        filename: str = "diffraction_grating_cells.stl",
        Q: int = 200,
        w: float = 0.1,
        cell_shape: str = "box",  # "box" или "cylinder"
) -> trimesh.Trimesh:
    """
    Формирует решетку ячеек в STL-формате.
    Можно выбрать форму ячейки: "box" (квадрат) или "cylinder".
    Размер ячейки берется из length_x.
    """

    a_np = np.asarray(a_params, dtype=float)
    h_np = np.asarray(h_params, dtype=float)

    list_drt, list_crt = get_drt_midrt(rt_np, orders)
    y0 = -0.0001 + 2 * stack * w

    rects = []  # (x_center_global, length_x, height)

    for m in range(orders):
        a = np.clip(a_np[m], 1e-3, None)

        if m == 0:
            rt0 = 0.0
        else:
            rt0 = rt_np[m - 1] + 0.01
        x0 = 0.0 if m == 0 else (rt_np[m] + rt0) / 2.0

        x = np.linspace(rt0, rt_np[m] - 0.01, Q)
        z = np.clip((x - x0) / a, -20.0, 20.0)
        y_profile = a * np.cosh(z)

        pure_x_list = []
        for n in range(stack, -1, -1):
            ip = int(np.argmin(y_profile))
            y_shifted = y_profile - y_profile[ip]
            ymax = 0.5 * (x.max() - x.min())
            y_top = np.clip(y_shifted, -ymax, ymax)
            y_bot = np.clip(y_shifted - w, -ymax, ymax)

            xx = np.concatenate([x, x[::-1]])
            yy = np.concatenate([y_top, y_bot[::-1]]) + 2.0 * n * w

            x_int = cut_y(xx, yy, y0)
            pure_x_list.append(x_int)

        filtered = []
        for arr in pure_x_list:
            if arr is None or arr.size == 0:
                continue
            L = arr.size
            if L == 1:
                arr = np.array([-arr[0], arr[0]], dtype=float)
            elif L == 2 and m == 0:
                v0, v1 = float(arr[0]), float(arr[1])
                arr = np.array([-v1, -v0, v0, v1], dtype=float)
            elif L in (2, 4):
                arr = arr - list_crt[m]
            else:
                continue
            filtered.append(arr)

        for arr in filtered:
            if arr.size == 2:
                intervals = [(float(arr[0]), float(arr[1]))]
            elif arr.size == 3:
                intervals = [(float(arr[0]), float(arr[1])),
                             (-float(arr[1]), -float(arr[0]))]
            else:
                intervals = [(float(arr[0]), float(arr[1])),
                             (float(arr[2]), float(arr[3]))]

            for (l, r) in intervals:
                length_x = float(r) - float(l)
                if length_x <= LIMIT_SHAPE:
                    continue

                x_center_global = (float(r) + float(l)) / 2.0 + x0
                rects.append((x_center_global, length_x, float(h_np[m])))

    # --- строим геометрию ---
    meshes = []

    W = 20.0
    L = float(rt_np[-1]) * 2.0
    H = float(substrate_thickness)

    substrate = trimesh.creation.box(extents=(L, W, H))
    substrate.apply_translation((0.0, 0.0, H / 2.0))
    meshes.append(substrate)

    for x_center, length_x, height in rects:
        ny = int(W // (6*length_x))
        for iy in range(-ny, ny + 1):
            y_center = iy * (2*length_x)

            if cell_shape == "box":
                # квадратная ячейка
                cell = trimesh.creation.box(extents=(length_x, length_x, height))
            elif cell_shape == "cylinder":
                # цилиндр радиус = length_x/2, высота = height
                cell = trimesh.creation.cylinder(
                    radius=length_x / 2.0,
                    height=height,
                    sections=32  # гладкость цилиндра
                )
            else:
                raise ValueError("cell_shape должен быть 'box' или 'cylinder'")

            # смещаем ячейку
            cell.apply_translation((x_center, y_center, H + height / 2.0))
            meshes.append(cell)

            # зеркальная копия по X
            mirror_cell = cell.copy()
            mirror_cell.apply_transform(np.diag([-1, 1, 1, 1]))
            meshes.append(mirror_cell)

    if meshes:
        combined = trimesh.util.concatenate(meshes)
        combined.export(filename)
        print(f"STL сохранён: {filename} (ячейки: {cell_shape})")
        return combined
    else:
        print("Нет геометрии для экспорта в STL")
        return None

def airy_characteristics():
    """
    Calculates the characteristic dimensions for an ideal focal spot (Airy disk).
    :return: dict with FWHM, FWTM, diameter of the first minimum
    """
    fwhm = 0.514 * conf.inc_wavelength / conf.numerical_aperture
    fwtm = 0.885 * conf.inc_wavelength / conf.numerical_aperture
    airy_diameter = 1.22 * conf.inc_wavelength / conf.numerical_aperture
    print(f"FWHM: {fwhm:.2f} um , FWTM: {fwtm:.2f} um, Airy diameter: {airy_diameter:.2f} um")