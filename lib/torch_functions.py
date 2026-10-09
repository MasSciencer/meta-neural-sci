import torch
import matplotlib.pyplot as plt
import numpy as np


def calculate_radii(D, f, wavelength, orders, split):
    """
    Calculates the radii on the lens surface at which a complete phase shift occurs from 0 to 2π,
    with the ability to split the first radius into several parts.

    :arg:
        D: The diameter of the lens (scalar or torch.Tensor)
        f: The focal length of the lens (scalar or torch.Tensor)
        wavelength: The wavelength of light (scalar or torch.Tensor)
        orders: The maximum value of n for calculating radii (int)
        split: How many parts to divide the first radius r[0] (int). If split <= 1, there is no split.

    :return:
        torch.Tensor (float64): A tensor of radii satisfying the condition r <= D/2,
        ordered in ascending order, taking into account the layout of the first radius.
    """
    dtype = torch.float64
    device = f.device if isinstance(f, torch.Tensor) else None

    D = torch.as_tensor(D, dtype=dtype, device=device)
    f = torch.as_tensor(f, dtype=dtype, device=device)
    wavelength = torch.as_tensor(wavelength, dtype=dtype, device=device)

    # creating a tensor of indexes n from 1 to orders
    n = torch.arange(1, orders + 1, dtype=dtype, device=device)

    # calculate all possible radii using the formula
    r_squared = n * wavelength**2 + 2 * n * wavelength * f
    radii = torch.sqrt(r_squared)

    # mask of radii falling into the lens aperture
    mask = radii <= (D / 2)
    if not torch.all(mask):
        print(f"Warning: {torch.sum(~mask).item()} the radii were discarded because they exceed D/2.")
    valid_radii = radii[mask]

    # If you need to split the first radius into parts, add them before valid_radii.
    if split and split > 1 and valid_radii.numel() > 0:
        r0 = valid_radii[0]
        # make split-1 points: r0/split, 2*r0/split, ..., (split-1)*r0/split
        subdiv = torch.arange(1, split, dtype=dtype, device=device) * r0 / split
        # Concatenate the results.
        valid_radii = torch.cat((subdiv, valid_radii), dim=0)

    return valid_radii

def calculate_radii_step(D, orders):
    """
    Calculates the set of radii obtained by evenly dividing the radius of the lens into orders of parts.

        :arg:
            D: The diameter of the lens (scalar or torch.Tensor)
            orders: The number of radius segments (int or scalar tensor)

        :return:
            The radius tensor (float64), ordered in ascending order,
            The dimension is [orders], in the range (0, D/2].
        """
    dtype = torch.float64
    device = D.device if isinstance(D, torch.Tensor) else None

    D = torch.as_tensor(D, dtype=dtype, device=device)
    orders_int = int(orders) if not isinstance(orders, torch.Tensor) else int(orders.item())

    # full radius
    r_max = D / 2

    # indexes from 1 to orders_int
    n = torch.arange(1, orders_int + 1, dtype=dtype, device=device)

    # dividing the radius into equal parts: r_i = (i / orders_int) * r_max
    radii = n * (r_max / orders_int)

    return radii


def get_midrt(rt, orders):
    """
    Calculate drt and midrt tensors from rt tensor.

    Args:
        rt: 1D torch.Tensor of values
        orders: int, number of orders to process

    Returns:
        tuple of (drt, midrt) torch.Tensors
    """
    midrt = torch.zeros(orders, dtype=rt.dtype, device=rt.device)
    midrt[0] = 0

    # Remaining elements
    if orders > 1:
        midrt[1:] = (rt[1:orders] + rt[:orders - 1]) / 2

    return midrt


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
    """Initializing a plot for visualization"""
    global fig, ax
    fig, ax = plt.subplots(figsize=(50, 10))
    # Defining the color palette
    global colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_curve(x, y, X_sections, y0, n):
    """
    Function for visualizing the curve and intersection points
    :param xx: x-coordinates of the curve
    :param yy: y-coordinates of the curve
    :param X_sections: points of intersection with the horizontal line
    :param y0: y-coordinate of the horizontal line
    :param n: curve number (for color selection)
    """

    x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

    # Plot the curve.
    ax.plot(x, y, color=colors[n % len(colors)], linewidth=0.5)

    # Highlight intersections in red.
    ax.scatter(X_sections, np.full_like(X_sections, y0), color='red', s=10, zorder=3)

def material_grid_weight_1d(
        a_params: torch.Tensor,  # shape = (orders,)
        h_params: torch.Tensor,  # shape = (orders,) - absolute material height in micrometers
        height_limit: int,  # list of upper height limits (micrometers) per order
        stack: int,
        rt_tf: torch.Tensor,
        drt_tf: torch.Tensor,
        resolutions: list,  # list of tuples (N_x, N_y) per order
        Q: int = 200,
        w: float = 0.1,
        smooth: bool = True,
        beta: float = 1.0,
        LIMIT_SHAPE: float = 0.2,
        device: torch.device = None,
        plot: bool = True,
) -> list:
    """
    A variant of the function without height optimization.
    """
    if device is None:
        device = a_params.device

    midrt_tf = get_midrt(rt_tf, len(a_params))
    y0 = -0.0001 + 2 * stack * w
    if plot:
        setup_visualization()
        ax.axhline(y=y0, color='black', linestyle='--', linewidth=1)

    result_weights = []
    for m in range(len(a_params)):
        # Parameters.
        a = torch.clamp(a_params[m], min=1e-3)
        h = torch.clamp(h_params[m], min=1e-3)
        N_x, N_y = resolutions[m]

        # meshes
        rt0 = torch.zeros((), device=device) if m == 0 else rt_tf[m-1]
        x0 = torch.zeros((), device=device) if m == 0 else (rt_tf[m] + rt0) / 2
        x = torch.linspace(rt0, rt_tf[m], Q, device=device)

        # Profile along x.
        z = torch.clamp((x - x0) / a, min=-20, max=20)
        y_profile = a * torch.cosh(z)

        # Collect intersections.
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

        # normalization of points
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
            elif L in (2,4):
                arr = arr - midrt_tf[m]
            else:
                continue
            filtered.append(arr)

        result_h = torch.zeros(N_y, device=device, dtype=h_params.dtype)
        result_x = torch.zeros(N_x, device=device, dtype=h_params.dtype)
        result_2d = torch.zeros((N_y, N_x), device=device, dtype=a_params.dtype)
        # global coordinates
        global_x = torch.linspace(-0.5 * drt_tf[m], 0.5 * drt_tf[m], N_x, device=device)
        # cell height: centers from 0 to h_limit
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
                # weights on x
                if smooth:
                    # x
                    dist_x = torch.clamp(start - global_x, min=0) + torch.clamp(global_x - end, min=0)
                    arg_x = torch.clamp(beta * dist_x, max=400)
                    mask_x = torch.sigmoid(-arg_x) * 2
                    result_x = torch.max(result_x, mask_x)
                    mask_h = global_h <= h
                    result_h = result_h.clone()  # Create a copy.
                    result_h = result_h.masked_fill(mask_h, 1.0)
                else:
                    mask_x = ((global_x >= start) & (global_x <= end))
                    result_x[mask_x] = 1.0
                    mask_h = global_h <= h
                    result_h[mask_h] = 1.0

                # combining x with the mask h
                # result = max(prev, mask_x[:,None] * mask_h[None,:])
                # result_x.unsqueeze(0)  # (1, N_x)
                # result_h.unsqueeze(1)  # (N_y, 1) This dimension should be the first one to be correctly correlated with the simulation.

                result_2d = torch.max(
                    result_2d,
                    result_h.unsqueeze(1)* result_x.unsqueeze(0)
                )

        result_weights.append(result_2d)
    return result_weights  # list of tensors shape=(N_x, N_y) per order

def material_grid_weight_2d(
        a_params: torch.Tensor,     # shape = (orders,)
        h_params: torch.Tensor,     # shape = (orders,) - absolute material height in micrometers
        height_limit: int,          # list of upper height limits (micrometers) per order
        stack: int,
        rt_tf: torch.Tensor,
        drt_tf: torch.Tensor,
        resolutions: list,          # list of tuples (N_x, N_y) per order
        Q: int = 200,
        w: float = 0.1,
        cut_shift: float = 0.0,
        smooth: bool = True,
        beta: float = 1.0,
        LIMIT_SHAPE: float = 0.2,
        device: torch.device = None,
        plot: bool = True,
) -> list:
    """
    Returns a list of tensors of weights shape =(orders, N_x, N_y).
    For x, weights_x weights are applied (smoothed or not).
    By height (y), the material exists where y_pixel <= h_params[m],
    otherwise, 0; mask_h is differentiated in height from 0 to height_limits[m] by N_y.

    :param cut_shift [-1:1] adjust shift cross-section line

    """
    if device is None:
        device = a_params.device

    midrt_tf = get_midrt(rt_tf, len(a_params))
    y0 = -0.0001 + 2 * stack * w + cut_shift * w
    if plot:
        setup_visualization()
        ax.axhline(y=y0, color='black', linestyle='--', linewidth=1)
    result_weights = []
    for m in range(len(a_params)):
        # key params
        a = a_params[m]
        h = h_params[m]
        N_x, N_y = resolutions[m]

        # meshes
        rt0 = torch.zeros((), device=device) if m == 0 else rt_tf[m-1]
        x0 = torch.zeros((), device=device) if m == 0 else (rt_tf[m] + rt0) / 2
        x = torch.linspace(rt0, rt_tf[m], Q, device=device)

        # x profile
        z = torch.clamp((x - x0) / a, min=-20, max=20)
        y_profile = a * torch.cosh(z)

        # collecting intersections
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

        # normalization of points
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
            elif L in (2,4):
                arr = arr - midrt_tf[m]
            else:
                continue
            filtered.append(arr)

        result_h = torch.zeros(N_y, device=device, dtype=h_params.dtype)
        result_x = torch.zeros(N_x, device=device, dtype=h_params.dtype)
        result_2d = torch.zeros((N_y, N_x), device=device, dtype=a_params.dtype)
        # global coordinates
        global_x = torch.linspace(-0.5 * drt_tf[m], 0.5 * drt_tf[m], N_x, device=device)
        # cell height: centers from 0 to h_limit
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
                # weights on x
                if smooth:
                    # x
                    dist_x = torch.clamp(start - global_x, min=0) + torch.clamp(global_x - end, min=0)
                    arg_x = torch.clamp(beta * dist_x, max=400)
                    mask_x = torch.sigmoid(-arg_x) * 2
                    result_x = torch.max(result_x, mask_x)
                    # y
                    dist_h = torch.clamp(global_h - h, min=0.0)
                    # we scale through beta and do not let the exponent overflow
                    arg_h = torch.clamp(beta*3 * dist_h, max=400)
                    # sigmoid turns a "soft" rectangle into [0,1]
                    # when global_y << h, arg_h≈0, sigmoid(0)=0.5, mask_h_soft≈0.5
                    # can be shifted by +0.5 if you need a single plateau (see below)
                    mask_h_soft = torch.sigmoid(-arg_h)
                    mask_h = mask_h_soft * 2 # to get plateau=1 inside and a smooth descent to 0
                    result_h = torch.clamp(mask_h, max=1.0)
                else:
                    mask_x = ((global_x >= start) & (global_x <= end))
                    result_x[mask_x] = 1.0
                    mask_h = global_h <= h
                    result_h[mask_h] = 1.0

                # combining x with the mask h
                # result = max(prev, mask_x[:,None] * mask_h[None,:])
                # result_x.unsqueeze(0)  # (1, N_x)
                # result_h.unsqueeze(1)  # (N_y, 1) This dimension should be the first one to be correctly correlated with the simulation.

                result_2d = torch.max(
                    result_2d,
                    result_h.unsqueeze(1)* result_x.unsqueeze(0)
                )
        result_weights.append(result_2d)
    return result_weights  # list of tensors shape=(N_x, N_y) per order

# def material_grid_weight_2d_rotate(
#         a_params: torch.Tensor,     # shape = (orders,)
#         h_params: torch.Tensor,     # shape = (orders,) - absolute material height in micrometers
#         height_limit: int,          # list of upper height limits (micrometers) per order
#         stack: int,
#         rt: torch.Tensor,
#         resolutions: list,          # list of tuples (N_x, N_y) per order
#         Q: int = 200,
#         w: float = 0.1,
#         cut_shift: float = 0.0,
#         smooth: bool = True,
#         beta: float = 1.0,
#         LIMIT_SHAPE: float = 0.2,
#         device: torch.device = None,
#         plot: bool = True,
# ) -> list:
#     """
#     Returns a list of tensors of weights shape =(orders, N_x, N_y).
#     For x, weights_x weights are applied (smoothed or not).
#     By height (y), the material exists where y_pixel <= h_params[m],
#     otherwise, 0; mask_h is differentiated in height from 0 to height_limits[m] by N_y.
#
#     :param cut_shift [-1:1] adjust shift cross-section line
#
#     """
#     if device is None:
#         device = a_params.device
#
#     list_drt, list_crt = get_drt_midrt(rt, len(a_params))
#     y0 = -0.01 + 2 * stack * w + w * cut_shift
#     if plot:
#         setup_visualization()
#         ax.axhline(y=y0, color='black', linestyle='--', linewidth=1)
#     result_weights = []
#     for m in range(len(a_params)):
#         # key params
#         a = a_params[m]
#         h = h_params[m]
#         N_x, N_y = resolutions[m]
#
#         # meshes
#         rt0 = torch.zeros((), device=device) if m == 0 else rt[m-1] + 0.01
#         x0 = torch.zeros((), device=device) if m == 0 else (rt[m] + rt0) / 2
#         x = torch.linspace(rt0, rt[m], Q, device=device)
#
#         # x profile
#         z = torch.clamp((x - x0) / a, min=-20, max=20)
#         y_profile = a * torch.cosh(z)
#
#         # collecting intersections
#         pure_x_list = []
#         for n in range(stack, -1, -1):
#
#             # ----- Base curves -----
#             ip = torch.argmin(y_profile)
#             y_shifted = y_profile - y_profile[ip]
#             ymax = 0.5 * (x.max() - x.min())
#             y_top = torch.clamp(y_shifted, -ymax, ymax)
#             y_bot = torch.clamp(y_shifted - w, -ymax, ymax)
#
#             xx = torch.cat([x, x.flip(0)], dim=0)
#             yy = torch.cat([y_top, y_bot.flip(0)], dim=0)
#
#             # =========================================================
#             #     1. LINEAR SHIFT
#             # =========================================================
#             dy = n * w
#             xx_shift = xx
#             yy_shift = yy + dy
#
#             # =========================================================
#             #     2. ANGULAR OFFSET FOR (n > 0)
#             # =========================================================
#
#             if m == 0:
#                 # Disable rotation for the first sector.
#                 xx_rot = xx
#                 yy_rot = yy + 2 * n * w
#             else:
#                 # Absolute radius of the sector center.
#                 r = list_crt[m]
#
#                 # Base angular offset (scalar).
#                 base_theta = w / (r + 1e-9)
#                 base_theta = torch.as_tensor(base_theta, device=device)
#
#                 # Rotation angle depends on the line index n.
#                 theta = n * base_theta
#
#                 cos_t = torch.cos(theta)
#                 sin_t = torch.sin(theta)
#
#                 # Rotate around the global center.
#                 xx_rot = xx_shift * cos_t - yy_shift * sin_t
#                 yy_rot = xx_shift * sin_t + yy_shift * cos_t
#
#             x_int = cut_y(xx_rot, yy_rot, y0)
#             pure_x_list.append(x_int)
#
#             if plot:
#                 plot_curve(xx_rot, yy_rot, x_int, y0, n)
#
#         # normalization of points
#         filtered = []
#         for arr in pure_x_list:
#             L = arr.numel()
#             if L == 0:
#                 continue
#             if L == 1:
#                 arr = torch.stack([-arr[0], arr[0]])
#             elif L == 2 and m == 0:
#                 v0, v1 = arr
#                 arr = torch.stack([-v1, -v0, v0, v1])
#             elif L in (2,4):
#                 arr = arr - list_crt[m]
#             else:
#                 continue
#             filtered.append(arr)
#
#         result_h = torch.zeros(N_y, device=device, dtype=h_params.dtype)
#         result_x = torch.zeros(N_x, device=device, dtype=h_params.dtype)
#         result_2d = torch.zeros((N_y, N_x), device=device, dtype=a_params.dtype)
#         # global coordinates
#         global_x = torch.linspace(-0.5 * list_drt[m], 0.5 * list_drt[m], N_x, device=device)
#         # cell height: centers from 0 to h_limit
#         global_h = torch.linspace(0, height_limit, N_y, device=device)
#
#         for arr in filtered:
#             if arr.numel() == 2:
#                 intervals = [(arr[0], arr[1])]
#             elif arr.numel() == 3:
#                 intervals = [(arr[0], arr[1]), (-arr[1], -arr[0])]
#             else:
#                 intervals = [(arr[0], arr[1]), (arr[2], arr[3])]
#
#             for start, end in intervals:
#                 if (end - start).abs() < LIMIT_SHAPE:
#                     continue
#                 # weights on x
#                 if smooth:
#                     # x
#                     dist_x = torch.clamp(start - global_x, min=0) + torch.clamp(global_x - end, min=0)
#                     arg_x = torch.clamp(beta * dist_x, max=400)
#                     mask_x = torch.sigmoid(-arg_x) * 2
#                     result_x = torch.max(result_x, mask_x)
#                     # y
#                     dist_h = torch.clamp(global_h - h, min=0.0)
#                     # we scale through beta and do not let the exponent overflow
#                     arg_h = torch.clamp(beta*3 * dist_h, max=400)
#                     # sigmoid turns a "soft" rectangle into [0,1]
#                     # when global_y << h, arg_h≈0, sigmoid(0)=0.5, mask_h_soft≈0.5
#                     # can be shifted by +0.5 if you need a single plateau (see below)
#                     mask_h_soft = torch.sigmoid(-arg_h)
#                     mask_h = mask_h_soft * 2 # to get plateau=1 inside and a smooth descent to 0
#                     result_h = torch.clamp(mask_h, max=1.0)
#                 else:
#                     mask_x = ((global_x >= start) & (global_x <= end))
#                     result_x[mask_x] = 1.0
#                     mask_h = global_h <= h
#                     result_h[mask_h] = 1.0
#
#                 # combining x with the mask h
#                 # result = max(prev, mask_x[:,None] * mask_h[None,:])
#                 # result_x.unsqueeze(0)  # (1, N_x)
#                 # result_h.unsqueeze(1)  # (N_y, 1) This dimension should be the first one to be correctly correlated with the simulation.
#
#                 result_2d = torch.max(
#                     result_2d,
#                     result_h.unsqueeze(1)* result_x.unsqueeze(0)
#                 )
#         result_weights.append(result_2d)
#     return result_weights  # list of tensors shape=(N_x, N_y) per order

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
    Generates a torch.Dimension tensor (orders,) with coefficients,
    starting with the base_value and going up/down.

    :param orders: Number of orders (vector length).
    :param base_value: Value for order 0.
    :param bias: Change coefficient (step for linear mode, base for exponential mode).
    :param is_rise: True increases values with the index; False decreases them.
    :param mode: Either 'linear' or 'exponential'.
    :param min_value: Lower bound on the final values.
    :param max_value: Upper bound on the final values.
    :param output: Whether to plot the result.
    :return: torch.Tensor containing the resulting coefficients.
    """
    idx = torch.arange(orders, dtype=torch.float32)

    if mode == 'linear':
        # Linear: base_value +/- bias * index.
        vals = base_value + bias * idx if is_rise else base_value - bias * idx

    elif mode == 'exp':
        # Exponential: base_value * bias^index or base_value / bias^index.
        if is_rise:
            vals = base_value * (bias ** idx)
        else:
            vals = base_value * (bias ** (-idx))
    else:
        raise ValueError("the mode should be 'linear' or 'exponential'")

    # Apply the bounds.
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

def get_effective_lr(optimizer: torch.optim.Optimizer, verbose: bool = False) -> float:
    """
    Return the mean effective learning rate
    for an Adam or AdamW optimizer.
    Print values for all parameters when verbose=True.

    Formula:
        effective_lr = lr / (sqrt(v_hat) + eps)
    where v_hat is the bias-corrected exponential estimate of the gradient variance.
    """
    eff_lrs = []

    for group in optimizer.param_groups:
        lr = group['lr']
        beta2 = group['betas'][1]
        eps = group['eps']

        for p in group['params']:
            if p.grad is None:
                continue
            state = optimizer.state.get(p, None)
            if not state or 'exp_avg_sq' not in state:
                continue

            v = state['exp_avg_sq']
            step_num = state['step']
            if step_num == 0:
                continue

            bias_correction2 = 1 - beta2 ** step_num
            denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

            eff_lr = (lr / denom).mean().item()
            eff_lrs.append(eff_lr)

            if verbose:
                print(f"  Param id={id(p)} | Effective LR = {eff_lr:.6e}")

    return float(sum(eff_lrs) / len(eff_lrs)) if eff_lrs else 0.0