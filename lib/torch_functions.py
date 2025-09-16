import torch
import matplotlib.pyplot as plt
import numpy as np


def calculate_radii(D, f, wavelength, orders, split):
    """
    Вычисляет радиусы на поверхности линзы, на которых происходит полный фазовый сдвиг от 0 до 2π,
    с возможностью разбить первый радиус на несколько частей.

    Args:
        D: Диаметр линзы (скаляр или torch.Tensor)
        f: Фокусное расстояние линзы (скаляр или torch.Tensor)
        wavelength: Длина волны света (скаляр или torch.Tensor)
        orders: Максимальное значение n для расчета радиусов (int)
        split: На сколько частей разделить первый радиус r[0] (int).
               Если split <= 1, разделения не происходит.

    Returns:
        torch.Tensor (float64): Тензор радиусов, удовлетворяющих условию r <= D/2,
        упорядоченный по возрастанию, с учётом раскладки первого радиуса.
    """
    dtype = torch.float64
    device = f.device if isinstance(f, torch.Tensor) else None

    # Приведение к нужному типу и устройству
    D = torch.as_tensor(D, dtype=dtype, device=device)
    f = torch.as_tensor(f, dtype=dtype, device=device)
    wavelength = torch.as_tensor(wavelength, dtype=dtype, device=device)

    # Создаём тензор индексов n от 1 до orders
    n = torch.arange(1, orders + 1, dtype=dtype, device=device)

    # Вычисляем все возможные радиусы по формуле
    r_squared = n * wavelength**2 + 2 * n * wavelength * f
    radii = torch.sqrt(r_squared)

    # Маска радиусов, попадающих в апертуру линзы
    mask = radii <= (D / 2)
    if not torch.all(mask):
        print(f"Предупреждение: {torch.sum(~mask).item()} радиусов были отброшены, так как превышают D/2.")

    # Применяем маску
    valid_radii = radii[mask]

    # Если нужно разбить первый радиус на части — добавляем их перед valid_radii
    if split and split > 1 and valid_radii.numel() > 0:
        r0 = valid_radii[0]
        # создаём split-1 точек: r0/split, 2*r0/split, ..., (split-1)*r0/split
        subdiv = torch.arange(1, split, dtype=dtype, device=device) * r0 / split
        # объединяем
        valid_radii = torch.cat((subdiv, valid_radii), dim=0)

    return valid_radii

def calculate_radii_step(D, orders):
    """
    Вычисляет набор радиусов, получающихся при равномерном делении радиуса линзы на orders частей.

    Args:
        D: Диаметр линзы (скаляр или torch.Tensor)
        orders: Количество сегментов радиуса (int или скалярный tensor)

    Returns:
        Тензор радиусов (float64), упорядоченный по возрастанию,
        размерностью [orders], в диапазоне (0, D/2].
    """
    dtype = torch.float64
    # Если D — tensor, запомним устройство, иначе defaults to CPU
    device = D.device if isinstance(D, torch.Tensor) else None

    # Приводим D к tensor
    D = torch.as_tensor(D, dtype=dtype, device=device)
    # orders нам нужен как целое число для arange
    orders_int = int(orders) if not isinstance(orders, torch.Tensor) else int(orders.item())

    # Полный радиус
    r_max = D / 2

    # Индексы от 1 до orders_int
    n = torch.arange(1, orders_int + 1, dtype=dtype, device=device)

    # Разбиваем радиус на равные части: r_i = (i / orders_int) * r_max
    radii = n * (r_max / orders_int)

    return radii


def get_drt_midrt(rt, orders):
    """
    Calculate drt and midrt tensors from rt tensor.

    Args:
        rt: 1D torch.Tensor of values
        orders: int, number of orders to process

    Returns:
        tuple of (drt, midrt) torch.Tensors
    """
    # Initialize tensors with zeros of the same dtype and device as rt
    drt = torch.zeros(orders, dtype=rt.dtype, device=rt.device)
    midrt = torch.zeros(orders, dtype=rt.dtype, device=rt.device)

    # First element special cases
    drt[0] = 2 * rt[0]
    midrt[0] = 0

    # Remaining elements
    if orders > 1:
        drt[1:] = rt[1:orders] - rt[:orders - 1]
        midrt[1:] = (rt[1:orders] + rt[:orders - 1]) / 2

    return drt, midrt


def cut_y(xx, yy, y0):
    """
    Find x-intercepts where yy crosses y0.

    Args:
        xx: 1D torch.Tensor of x-values
        yy: 1D torch.Tensor of y-values
        y0: float or scalar torch.Tensor, the y-value to find crossings

    Returns:
        torch.Tensor of x-values where yy crosses y0, sorted in ascending order
    """
    # Calculate differences from y0
    y_diff = yy - y0

    # Find where the sign changes (crossings)
    sign_changes = y_diff[1:] * y_diff[:-1] <= 0

    # Get indices where crossings occur
    crossing_indices = torch.nonzero(sign_changes, as_tuple=False).squeeze(-1)

    # Handle case where there are no crossings
    if crossing_indices.numel() == 0:
        return torch.tensor([], dtype=xx.dtype, device=xx.device)

    # Calculate exact x-intercepts using linear interpolation
    x1 = xx[crossing_indices]
    x2 = xx[crossing_indices + 1]
    y1 = yy[crossing_indices]
    y2 = yy[crossing_indices + 1]

    x_intersections = x1 + (y0 - y1) * (x2 - x1) / (y2 - y1)

    # Sort the intersections
    sorted_x = torch.sort(x_intersections).values

    return sorted_x

def setup_visualization():
    """Инициализация фигуры для визуализации"""
    global fig, ax
    fig, ax = plt.subplots(figsize=(10, 10))
    # Определяем палитру цветов
    global colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_curve(x, y, X_sections, y0, n):
    """
    Функция для визуализации кривой и точек пересечения
    :param xx: x-координаты кривой
    :param yy: y-координаты кривой
    :param X_sections: точки пересечения с горизонтальной линией
    :param y0: y-координата горизонтальной линии
    :param n: номер кривой (для выбора цвета)
    """

    x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

    # Рисуем кривую
    ax.plot(x, y, color=colors[n % len(colors)], linewidth=0.5)

    # Выделяем точки пересечения красным
    ax.scatter(X_sections, np.full_like(X_sections, y0), color='red', s=10, zorder=3)

def material_grid_weight_2d_rotate(
        a_params: torch.Tensor,     # shape = (orders,)
        h_params: torch.Tensor,     # shape = (orders,) - absolute material height in micrometers
        height_limit: int,          # list of upper height limits (micrometers) per order
        orders: int,
        stack: int,
        rt: torch.Tensor,
        resolutions: list,          # list of tuples (N_x, N_y) per order
        Q: int = 200,
        w: float = 0.1,
        smooth: bool = True,
        beta: float = 1.0,
        LIMIT_SHAPE: float = 0.2,
        device: torch.device = None,
        plot: bool = True,
) -> list:
    """
    Возвращает список тензоров весов shape = (orders, N_x, N_y).
    По x применяются веса weights_x (сглаженные или нет).
    По высоте (y) материал существует там, где y_pixel <= h_params[m],
    иначе 0; mask_h дифферируется по высоте от 0 до height_limits[m] разбивкой на N_y.
    h_params может превышать height_limits, тогда mask_h будет полностью единицами.
    """
    if device is None:
        device = a_params.device

    list_drt, list_crt = get_drt_midrt(rt, orders)
    y0 = -0.0001 + 2 * stack * w
    if plot:
        setup_visualization()
        ax.axhline(y=y0, color='black', linestyle='--', linewidth=1)

    result_weights = []
    for m in range(orders):
        a = torch.clamp(a_params[m], min=1e-3)
        h = torch.clamp(h_params[m], min=1e-3)
        N_x, N_y = resolutions[m]

        rt0 = torch.zeros((), device=device) if m == 0 else rt[m-1] + 0.01
        x0 = torch.zeros((), device=device) if m == 0 else (rt[m] + rt0) / 2

        x = torch.linspace(rt0, rt[m] - 0.01, Q, device=device)

        # профиль по x
        z = torch.clamp((x - x0) / a, min=-20, max=20)
        y_profile = a * torch.cosh(z)

        # сбор пересечений
        pure_x_list = []

        r = x0.abs().clamp(min=1e-6)  # чтобы не делить на ноль
        w_rad = (w / r).to(device)  # w и R должны быть в одних единицах

        for n in range(stack, -1, -1):
            theta = torch.tensor(n * w_rad, device=device)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            ip = torch.argmin(y_profile)
            y_shifted = y_profile - y_profile[ip]
            ymax = 0.1 * (x.max() - x.min())
            y_top = torch.clamp(y_shifted, -ymax, ymax)
            y_bot = torch.clamp(y_shifted - w, -ymax, ymax)

            xx = torch.cat([x, x.flip(0)], dim=0)
            yy = torch.cat([y_top, y_bot.flip(0)], dim=0) + 2 * n * w

            xx_local = xx - x0

            x_rot = xx_local * cos_t - yy * sin_t + x0 * cos_t
            y_rot = xx_local * sin_t + yy * cos_t + x0 * sin_t

            x_int = cut_y(x_rot, y_rot, y0)

            pure_x_list.append(x_int)

            if plot:
                plot_curve(x_rot, y_rot, x_int, y0, n)

        # нормировка
        filtered = []
        for arr in pure_x_list:
            L = arr.numel()
            if L == 0:
                continue
            if L == 1:
                arr = torch.stack([-arr[0], arr[0]])
            elif L == 2 and m == 0:
                v0, v1 = arr
                arr = torch.stack([-v1, -v0, v0, v1])
            elif L in (2,3,4):
                arr = arr - list_crt[m]
            else:
                continue
            filtered.append(arr)

        result_h = torch.zeros(N_y, device=device, dtype=h_params.dtype)
        result_x = torch.zeros(N_x, device=device, dtype=h_params.dtype)
        result_2d = torch.zeros((N_y, N_x), device=device, dtype=a_params.dtype)
        # глобальные координаты
        global_x = torch.linspace(-0.5 * list_drt[m], 0.5 * list_drt[m], N_x, device=device)
        # высота ячейки: центры от 0 до h_limit
        global_h = torch.linspace(0, height_limit, N_y, device=device)

        for arr in filtered:
            if arr.numel() == 2:
                intervals = [(arr[0], arr[1])]
            elif arr.numel() == 3:
                intervals = [(arr[0], arr[1]), (-arr[1], -arr[0])]
            else:
                intervals = [(arr[0], arr[1]), (arr[2], arr[3])]

            for start, end in intervals:
                if (end - start).abs() < LIMIT_SHAPE:
                    continue
                # веса по x
                if smooth:
                    # x
                    dist_x = torch.clamp(start - global_x, min=0) + torch.clamp(global_x - end, min=0)
                    arg_x = torch.clamp(beta * dist_x, max=400)
                    mask_x = torch.sigmoid(-arg_x) * 2
                    result_x = torch.max(result_x, mask_x)

                    # y
                    dist_h = torch.clamp(global_h - h, min=0.0)
                    # масштабируем через beta и не даём переполниться экспоненте
                    arg_h = torch.clamp(beta*3 * dist_h, max=400)
                    # sigmoid превращает «мягкий» прямоугольник в [0,1]
                    # когда global_y << h, arg_h≈0 ⇒ sigmoid(0)=0.5 ⇒ mask_h_soft≈0.5
                    mask_h_soft = torch.sigmoid(-arg_h)
                    mask_h = mask_h_soft * 2 # чтобы получить plateau=1 внутри и плавный спуск к 0
                    result_h = torch.clamp(mask_h, max=1.0)
                else:
                    mask_x = ((global_x >= start) & (global_x <= end))
                    result_x[mask_x] = 1.0
                    mask_h = global_h <= h
                    result_h[mask_h] = 1.0

                # комбинируем x с маской h

                result_2d = torch.max(
                    result_2d,
                    result_h.unsqueeze(1)* result_x.unsqueeze(0)
                )

        result_weights.append(result_2d)
    return result_weights  # list of tensors shape=(N_x, N_y) per order
def material_grid_weight_2d(
        a_params: torch.Tensor,   # shape = (orders,)
        h_params: torch.Tensor,   # shape = (orders,) - absolute material height in micrometers
        height_limit: int,      # list of upper height limits (micrometers) per order
        orders: int,
        stack: int,
        rt: torch.Tensor,
        resolutions: list,        # list of tuples (N_x, N_y) per order
        Q: int = 200,
        w: float = 0.1,
        smooth: bool = True,
        beta: float = 1.0,
        LIMIT_SHAPE: float = 0.2,
        device: torch.device = None,
        plot: bool = True,
) -> list:
    """
    Возвращает список тензоров весов shape = (orders, N_x, N_y).
    По x применяются веса weights_x (сглаженные или нет).
    По высоте (y) материал существует там, где y_pixel <= h_params[m],
    иначе 0; mask_h дифферируется по высоте от 0 до height_limits[m] разбивкой на N_y.
    h_params может превышать height_limits, тогда mask_h будет полностью единицами.
    """
    if device is None:
        device = a_params.device

    list_drt, list_crt = get_drt_midrt(rt, orders)
    y0 = -0.0001 + 2 * stack * w
    if plot:
        setup_visualization()
        ax.axhline(y=y0, color='black', linestyle='--', linewidth=1)

    result_weights = []
    for m in range(orders):
        # параметры
        a = torch.clamp(a_params[m], min=1e-3)
        h = torch.clamp(h_params[m], min=1e-3)
        N_x, N_y = resolutions[m]

        # сетки
        rt0 = torch.zeros((), device=device) if m == 0 else rt[m-1] + 0.01
        x0 = torch.zeros((), device=device) if m == 0 else (rt[m] + rt0) / 2
        x = torch.linspace(rt0, rt[m] - 0.01, Q, device=device)

        # профиль по x
        z = torch.clamp((x - x0) / a, min=-20, max=20)
        y_profile = a * torch.cosh(z)

        # сбор пересечений
        pure_x_list = []
        for n in range(stack, -1, -1):
            ip = torch.argmin(y_profile)
            y_shifted = y_profile - y_profile[ip]
            ymax = 0.5 * (x.max() - x.min())
            y_top = torch.clamp(y_shifted, -ymax, ymax)
            y_bot = torch.clamp(y_shifted - w, -ymax, ymax)

            xx = torch.cat([x, x.flip(0)], dim=0)
            yy = torch.cat([y_top, y_bot.flip(0)], dim=0) + 2 * n * w
            x_int = cut_y(xx, yy, y0)
            pure_x_list.append(x_int)
            if plot:
                plot_curve(xx, yy, x_int, y0, n)

        # нормировка точек
        filtered = []
        for arr in pure_x_list:
            L = arr.numel()
            if L == 0:
                continue
            if L == 1:
                arr = torch.stack([-arr[0], arr[0]])
            elif L == 2 and m == 0:
                v0, v1 = arr
                arr = torch.stack([-v1, -v0, v0, v1])
            elif L in (2,3,4):
                arr = arr - list_crt[m]
            else:
                continue
            filtered.append(arr)

        result_h = torch.zeros(N_y, device=device, dtype=h_params.dtype)
        result_x = torch.zeros(N_x, device=device, dtype=h_params.dtype)
        result_2d = torch.zeros((N_y, N_x), device=device, dtype=a_params.dtype)
        # глобальные координаты
        global_x = torch.linspace(-0.5 * list_drt[m], 0.5 * list_drt[m], N_x, device=device)
        # высота ячейки: центры от 0 до h_limit
        global_h = torch.linspace(0, height_limit, N_y, device=device)

        for arr in filtered:
            if arr.numel() == 2:
                intervals = [(arr[0], arr[1])]
            elif arr.numel() == 3:
                intervals = [(arr[0], arr[1]), (-arr[1], -arr[0])]
            else:
                intervals = [(arr[0], arr[1]), (arr[2], arr[3])]

            for start, end in intervals:
                if (end - start).abs() < LIMIT_SHAPE:
                    continue
                # веса по x
                if smooth:
                    # x
                    dist_x = torch.clamp(start - global_x, min=0) + torch.clamp(global_x - end, min=0)
                    arg_x = torch.clamp(beta * dist_x, max=400)
                    mask_x = torch.sigmoid(-arg_x) * 2
                    result_x = torch.max(result_x, mask_x)

                    # y
                    dist_h = torch.clamp(global_h - h, min=0.0)
                    # масштабируем через beta и не даём переполниться экспоненте
                    arg_h = torch.clamp(beta*3 * dist_h, max=400)
                    # sigmoid превращает «мягкий» прямоугольник в [0,1]
                    # когда global_y << h, arg_h≈0 ⇒ sigmoid(0)=0.5 ⇒ mask_h_soft≈0.5
                    # можно сдвинуть на +0.5, если надо единичный plateau (см. ниже)
                    mask_h_soft = torch.sigmoid(-arg_h)
                    mask_h = mask_h_soft * 2 # чтобы получить plateau=1 внутри и плавный спуск к 0
                    result_h = torch.clamp(mask_h, max=1.0)
                else:
                    mask_x = ((global_x >= start) & (global_x <= end))
                    result_x[mask_x] = 1.0
                    mask_h = global_h <= h
                    result_h[mask_h] = 1.0

                # комбинируем x с маской h
                # result = max(prev, mask_x[:,None] * mask_h[None,:])
                # result_x.unsqueeze(0)  # (1, N_x)
                # result_h.unsqueeze(1)  # (N_y, 1) Эта размерность должна быть первой, чтобы правильно соотносится с симуляцией

                result_2d = torch.max(
                    result_2d,
                    result_h.unsqueeze(1)* result_x.unsqueeze(0)
                )

        result_weights.append(result_2d)

    return result_weights  # list of tensors shape=(N_x, N_y) per order

def material_grid_weight(
        a_params: torch.Tensor,  # shape = (orders,)
        orders: int,
        stack: int,
        rt: torch.Tensor,
        resolutions: list,
        Q: int = 200,
        w: float = 0.1,
        smooth: bool = True,
        beta: float = 1.0,
        LIMIT_SHAPE: float = 0.2,
        device: torch.device = None,
        plot=True,
) -> torch.Tensor:
    """
    Возвращает тензор весов shape = (orders, adaptive_resolution_i).
    Отрезки короче LIMIT_SHAPE обнуляются (не учитываются).
    """
    if device is None:
        device = a_params.device

    list_drt, list_crt = get_drt_midrt(rt, orders)
    y0 = (-0.0001 + 2 * stack * w)

    if plot:
        setup_visualization()
        ax.axhline(y=y0, color='black', linestyle='--', linewidth=1)

    result_weights = []

    for m in range(orders):
        a = a_params[m]
        rt0 = torch.zeros((), device=device) if m == 0 else rt[m - 1] + 0.01
        x0 = torch.zeros((), device=device) if m == 0 else (rt[m] + rt0) / 2

        x = torch.linspace(rt0, rt[m] - 0.01, Q, device=device)

        eps = 1e-3
        a = torch.clamp(a, min=eps)
        z = torch.clamp((x - x0) / a, min=-20, max=20.0)
        y = a * torch.cosh(z)

        pure_x_list = []
        for n in range(stack, -1, -1):
            ip = torch.argmin(y)
            y_shifted = y - y[ip]
            ymax = 0.5 * (x.max() - x.min())

            y_top = torch.clamp(y_shifted, -ymax, ymax)
            y_bot = torch.clamp(y_shifted - w, -ymax, ymax)

            xx = torch.cat([x, x.flip(0)], dim=0)
            yy = torch.cat([y_top, y_bot.flip(0)], dim=0) + 2 * n * w

            x_pure = cut_y(xx, yy, y0)
            pure_x_list.append(x_pure)

            if plot:
                plot_curve(xx, yy, x_pure, y0, n)

        filtered_x_list = []
        for idx, arr in enumerate(pure_x_list):
            L = arr.numel()
            if L == 0:
                continue
            elif L == 1:
                v = arr[0]
                arr_new = torch.stack([-v, v], dim=0)
            elif L == 2 and m == 0:
                v0, v1 = arr[0], arr[1]
                arr_new = torch.stack([-v1, -v0, v0, v1], dim=0)
            elif L in (2, 3, 4):
                arr_new = arr - list_crt[m]
            else:
                print(f"Unexpected number of intersection points: {L}")
                continue
            filtered_x_list.append(arr_new)

        x_list = filtered_x_list
        N = resolutions[m]
        global_vec = torch.linspace(-0.5 * list_drt[m], 0.5 * list_drt[m], N, device=device)
        result = torch.zeros(N, device=device, dtype=a_params.dtype)

        for arr in x_list:
            pts = arr.view(-1)
            # формируем список отрезков
            if pts.numel() == 2:
                intervals = [(pts[0], pts[1])]
            elif pts.numel() == 3:
                intervals = [(pts[0], pts[1]), (-pts[1], -pts[0])]
            else:  # 4
                intervals = [(pts[0], pts[1]), (pts[2], pts[3])]

            for start, end in intervals:
                # Фильтрация коротких отрезков
                if (end - start).abs() < LIMIT_SHAPE:
                    continue

                # вычисляем расстояния до отрезка
                dist_left = torch.clamp(start - global_vec, min=0.0)
                dist_right = torch.clamp(global_vec - end, min=0.0)
                distances = dist_left + dist_right

                if smooth:
                    arg = torch.clamp(beta * distances, max=400)
                    weights = torch.sigmoid(-arg) * 2
                    result = torch.max(result, weights)
                else:
                    mask = (global_vec >= start) & (global_vec <= end)
                    result[mask] = 1.0

        result_weights.append(result.unsqueeze(0))

    return result_weights


def auto_initial(
        orders: int,
        base_value: float,
        bias: float,
        is_rise: bool,
        mode: str = 'linear',
        min_value: float = 0.1,
        max_value: float = 100.0,
        output: bool = True,
) -> torch.Tensor:
    """
    Генерирует torch.Tensor размерности (orders,) с коэффициентами,
    начиная с base_value и далее растущими/падающими.

    :param orders: число порядков (длина вектора)
    :param base_value: значение для порядка 0
    :param bias: коэффициент изменения (шаг для linear, основание для exp)
    :param is_rise: если True — значения растут по мере роста индекса, иначе — убывают
    :param mode: 'linear' или 'exponential'
    :param min_value: нижняя граница финальных значений
    :param max_value: верхняя граница финальных значений
    :param output: если True — строит график
    :return: torch.Tensor с итоговыми коэффициентами
    """
    idx = torch.arange(orders, dtype=torch.float32)

    if mode == 'linear':
        # линейно: base_value ± bias * index
        vals = base_value + bias * idx if is_rise else base_value - bias * idx

    elif mode == 'exp':
        # экспоненциально: base_value * bias^index или / bias^index
        if is_rise:
            vals = base_value * (bias ** idx)
        else:
            vals = base_value * (bias ** (-idx))
    else:
        raise ValueError("mode должен быть 'linear' или 'exponential'")

    # применение границ
    vals = torch.clamp(vals, min=min_value, max=max_value)

    if output:
        plt.figure(figsize=(6, 4))
        plt.plot(idx.numpy(), vals.numpy(), marker='o')
        title_mode = 'linear' if mode == 'linear' else 'exponential'
        plt.title(f'auto_initial: {title_mode}, {"rise" if is_rise else "fall"}')
        plt.xlabel('Order (index)')
        plt.ylabel('Coefficients')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return vals
