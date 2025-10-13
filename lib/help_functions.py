import os
import numpy as np
import torch
import pickle
from typing import Union, Sequence


def get_gap_from_pinhole_to_surface(
        D_aperture: float,
        D_pinhole: float,
        max_angle_in_deg: float,
) -> float:

    max_angle_in_rad = np.radians(max_angle_in_deg)
    d_gap = (D_aperture - D_pinhole)/(2*np.tan(max_angle_in_rad))

    return d_gap

def get_max_angle_in_deg(D_aperture: float, D_pinhole: float, gap_from_pinhole_to_surface: float) -> float:
    max_angel = np.arctan ((D_aperture - D_pinhole) / (2 * gap_from_pinhole_to_surface))
    return np.rad2deg(max_angel)

def get_angle_list(n_angles: int, D_pinhole: float, gap_from_pinhole_to_surface: float):
    angle_list = []
    for i in range(n_angles):
        next_angle_in_rad = np.arctan ((i*D_pinhole) / (2 * gap_from_pinhole_to_surface))
        next_angle_in_deg = np.rad2deg(next_angle_in_rad)
        angle_list.append(next_angle_in_deg)
    return angle_list

def find_D_pinhole(
        n_angels: int,
        D_aperture: float,
) -> float:
    """
    Definition: Функция на основе дискретности оптимизируемых углов позволяет оценить диаметр диафрагмы

    :param n_angels:
    :param D_aperture:
    :return:
    """
    if n_angels <= 1:
        return D_aperture
    else:
        angle_orders = 2 * n_angels - 1
        d_pinhole = D_aperture / angle_orders

    return d_pinhole

def save_params(
    params: Union[torch.Tensor, Sequence[torch.Tensor]],
    name: str,
    np_dir: str = ".",
    pt_dir: str = "."
):
    """
    Сохранить тензор или список тензоров:

    - Если params — torch.Tensor, сохраняет его в name.npy и name.pt.
    - Если params — список/кортеж torch.Tensor, сохраняет весь список
      в name.npy (как object array) и name.pt.

    Args:
        params (torch.Tensor or Sequence[torch.Tensor]):
            тензор или список тензоров для сохранения.
        name (str): базовое имя файла (без расширения).
        np_dir (str): директория для .npy (по умолчанию текущая).
        pt_dir (str): директория для .pt (по умолчанию текущая).
    """
    os.makedirs(np_dir, exist_ok=True)
    os.makedirs(pt_dir, exist_ok=True)

    np_path = os.path.join(np_dir, f"{name}.npy")
    pt_path = os.path.join(pt_dir, f"{name}.pt")

    if isinstance(params, torch.Tensor):
        # --- Одиночный тензор ---
        tensor_cpu = params.detach().cpu()
        np_array = tensor_cpu.numpy()
        np.save(np_path, np_array)
        torch.save(params, pt_path)
        print(f"[Tensor] Saved NumPy array to {np_path}")
        print(f"[Tensor] Saved Torch tensor to {pt_path}")

    elif isinstance(params, (list, tuple)):
        # --- Список тензоров ---
        if not all(isinstance(t, torch.Tensor) for t in params):
            raise ValueError("Все элементы списка должны быть torch.Tensor")

        # Конвертируем каждый тензор в NumPy
        np_list = [t.detach().cpu().numpy() for t in params]
        # Упаковываем в object-array, чтобы numpy.save понял, что это разнородные формы
        obj_array = np.array(np_list, dtype=object)
        np.save(np_path, obj_array, allow_pickle=True)

        # Сохраняем оригинальный список тензоров
        torch.save(params, pt_path)

        print(f"[List]   Saved list of {len(params)} NumPy arrays to {np_path}")
        print(f"[List]   Saved list of {len(params)} Torch tensors to {pt_path}")

    else:
        raise TypeError("params must be a torch.Tensor or a list/tuple of torch.Tensor")

def save_history(loc_history=None, save=True, index=None):
    """
    Сохраняет или загружает словарь loc_history в файл.

    Параметры:
        loc_history (dict | None): словарь для сохранения (если save=True).
        save (bool): True -> сохранить, False -> загрузить.
        index (int | None): если указан, добавляется к имени файла ("loc_history_angle_{index}.pkl").

    Возвращает:
        str | dict: при сохранении возвращает имя файла, при загрузке — словарь.
    """
    filename = "loc_history.pkl" if index is None else f"loc_history_angle_{index}.pkl"

    if save:
        with open(filename, "wb") as f:
            pickle.dump(loc_history, f)
        print(f"[INFO] История сохранена в {filename}")
        return filename
    else:
        with open(filename, "rb") as f:
            loc_history = pickle.load(f)
        print(f"[INFO] История загружена из {filename}")
        return loc_history

def recenter_and_crop(whole_int: np.ndarray, abscissa: np.ndarray, ref_index: int):
    """
    Центрирует и симметрично обрезает IOptimized так, чтобы максимум выбранной длины волны оказался в центре.
    Возвращает обрезанный массив и соответствующий ему вектор abscissa.

    Parameters
    ----------
    IOptimized : np.ndarray
        Массив размером (n, m), где n - количество точек, m - количество длин волн.
    abscissa : np.ndarray
        Вектор длиной n, абсолютные координаты точек (например, в мкм).
    ref_index : int
        Индекс длины волны, используемой как эталон для центровки (столбец).

    Returns
    -------
    (np.ndarray, np.ndarray)
        - Новый массив (n', m), обрезанный и центрированный.
        - Новый вектор abscissa длиной n'.
    """
    n, m = whole_int.shape
    center = n // 2

    # индекс максимума у эталонной длины волны
    ref_max_idx = np.argmax(whole_int[:, ref_index])

    # величина сдвига
    shift = center - ref_max_idx

    # сдвигаем данные
    shifted = np.roll(whole_int, shift, axis=0)
    abscissa_shifted = np.roll(abscissa, shift)

    # считаем, сколько точек нужно обрезать с каждой стороны
    cut = abs(shift)
    if cut > 0:
        shifted = shifted[cut:-cut, :]
        abscissa_shifted = abscissa_shifted[cut:-cut]
        print(f"[INFO] Центровка: максимум был на {ref_max_idx}, сдвиг = {shift}. "
              f"Обрезано {cut} точек с каждой стороны. Итоговая длина = {len(abscissa_shifted)}")
    else:
        print(f"[INFO] Центровка: максимум уже в центре ({ref_max_idx}). Обрезка не требуется.")

    return shifted, abscissa_shifted

