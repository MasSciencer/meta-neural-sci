import numpy as np
import meep as mp
from lib import adjoint as mpa
from lib import torch_functions as tf
from lib import numpy_functions as nf
from lib import help_functions
from matplotlib import pyplot as plt
from matplotlib import animation
import conf
import torch

def init_loc_opt(index_angle, loc_orders, obj_func, a_initial, h_initial, rt_np, resolutions, stack_batch_src, locNearRegion, is_plot=False):
    """
    The function of starting a local cell
    """

    if index_angle == 0:
        extended_loc_orders = int(loc_orders / 2)
        is_global = True
        first_index = 0
        loc_symmetry = [mp.Mirror(mp.Y)]
    else:
        extended_loc_orders = loc_orders - 1
        is_global = False
        first_index = 1
        loc_symmetry = None

    print(f"extended_loc_orders: {extended_loc_orders}")

    loc_a_params = torch.full((extended_loc_orders,), a_initial, dtype=torch.float64)
    loc_h_params = torch.full((extended_loc_orders,), h_initial, dtype=torch.float64)

    loc_d_rt_np, loc_mid_rt_np = nf.get_drt_midrt(rt_np, extended_loc_orders, is_global=is_global)
    loc_rt_tf = tf.calculate_radii_step(D=conf.diaphragm, orders=extended_loc_orders)

    loc_focus_point_list = []
    point1 = mp.Vector3(x=conf.f, y=0)
    loc_focus_point_list.append(point1)

    loc_design_region = []
    list_loc_x0 = tf.material_grid_weight_2d(loc_a_params, loc_h_params, conf.at_hei, extended_loc_orders, conf.stack,
                                             loc_rt_tf, resolutions, Q=conf.Q, w=conf.w,
                                             LIMIT_SHAPE=conf.LIMIT_SHAPE, smooth=False, beta=50.0, plot=False)

    geometry = [mp.Block(
        material=conf.in_material,
        size=mp.Vector3(conf.in_sub, mp.inf, 0),
        center=mp.Vector3(conf.sub_center.x),
    )]

    for loc_order in range(first_index, extended_loc_orders):
        weight_numpy = list_loc_x0[
            loc_order].detach().cpu().numpy()  # тут вытаскиваем значения для первичной инициализации и для визуальной проверки
        res = resolutions[loc_order]
        material_grid = mp.MaterialGrid(
            mp.Vector3(x=res[1], y=res[0]),
            mp.air,
            conf.at_material,
            weights=weight_numpy
        )
        loc_design_region.append(mpa.DesignRegion(
            material_grid,
            volume=mp.Volume(center=mp.Vector3(x=conf.at_hei / 2, y=loc_mid_rt_np[loc_order]),
                             size=mp.Vector3(x=conf.at_hei, y=loc_d_rt_np[loc_order] - 0.1))
            # 0.1 slightly offset from the edge of the geometry so that the orders do not run into each other
        ))

        geometry.append(mp.Block(
            size=mp.Vector3(x=conf.at_hei, y=loc_d_rt_np[loc_order]),
            center=mp.Vector3(x=conf.at_hei / 2, y=loc_mid_rt_np[loc_order]),
            material=material_grid,
        ))

    loc_sim = mp.Simulation(
        cell_size=conf.loc_cell,
        boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.X, side=mp.ALL)],
        geometry=geometry,
        sources=stack_batch_src[index_angle],
        default_material=mp.Medium(index=1.0),
        resolution=conf.sim_resolution,
        symmetries=loc_symmetry,
    )

    FarFields = mpa.Near2FarFields(loc_sim, [locNearRegion], loc_focus_point_list)
    ob_list = [FarFields]

    loc_opt = mpa.OptimizationProblem(
        simulation=loc_sim,
        objective_functions=[obj_func],
        objective_arguments=ob_list,  # ready
        design_regions=loc_design_region,  # ready
        frequencies=conf.inc_frequents_array,  # ready
        maximum_run_time=conf.maximum_run_time,
        until_steps=conf.maximum_run_time,
    )

    if is_plot:
        plt.figure(figsize=(5, 100))
        loc_opt.plot2D(True)

    return loc_opt


def init_loc_opt_hyper(index_angle, loc_orders, a_initial, h_initial, p_min, p_max, h_min, h_max, lr, regime):
    if index_angle == 0:
        extended_loc_orders = int(loc_orders / 2)
    else:
        extended_loc_orders = loc_orders - 1

    # --- a_initial ---
    if isinstance(a_initial, (float, int)):
        loc_a_params = torch.full((extended_loc_orders,), a_initial, dtype=torch.float64)
    elif isinstance(a_initial, torch.Tensor):
        loc_a_params = a_initial
    else:
        raise TypeError("a_initial must be float, int, or torch.Tensor")

    # --- h_initial ---
    if isinstance(h_initial, (float, int)):
        loc_h_params = torch.full((extended_loc_orders,), h_initial, dtype=torch.float64)
    elif isinstance(h_initial, torch.Tensor):
        loc_h_params = h_initial
    else:
        raise TypeError("h_initial must be float, int, or torch.Tensor")
    # --- нормируем и инвертируем sigmoid ---
    loc_u_a0 = torch.log((loc_a_params - p_min) / (p_max - loc_a_params))

    loc_optimizer = None
    loc_u_a = None
    loc_u_h = None

    if regime==2:
        loc_u_h0 = torch.log((loc_h_params - h_min) / (h_max - loc_h_params))
        loc_u_a = torch.nn.Parameter(loc_u_a0)
        loc_u_h = torch.nn.Parameter(loc_u_h0)

        loc_optimizer = torch.optim.Adam(
            [loc_u_a, loc_u_h],
            lr=lr,
            betas=(0.95, 0.999)
        )
    elif regime==1.5:
        scalar_h0 = torch.tensor(h_initial, dtype=torch.float64)
        loc_u_h0 = torch.log((scalar_h0 - h_min) / (h_max - scalar_h0))
        loc_u_a = torch.nn.Parameter(loc_u_a0)
        init_u_h = torch.nn.Parameter(loc_u_h0)  # скаляр-параметр
        loc_u_h = init_u_h.repeat(extended_loc_orders)
        loc_optimizer = torch.optim.Adam(
            [loc_u_a, init_u_h],
            lr=lr,
            betas=(0.95, 0.999)
        )
    elif regime==1:
        loc_u_h0 = torch.log((loc_h_params - h_min) / (h_max - loc_h_params))
        loc_u_a = torch.nn.Parameter(loc_u_a0)
        loc_u_h = loc_u_h0
        loc_optimizer = torch.optim.Adam(
            [loc_u_a],
            lr=lr,
            betas=(0.95, 0.999)
        )
    else:
        ValueError("regime must be 1, 2 or 1.5")

    # История
    store_loc_history = {
        'f0': [],
        'grad_norm': [],
        'loc_a_params': [],
        'loc_h_params': [],
        'beta': []
    }

    return loc_optimizer, store_loc_history, loc_u_a, loc_u_h


def run_loc_opt(opt_torch, index_angle, loc_orders, loc_optimizer, loc_u_a, loc_u_h, loc_history, beta, num_steps, decoder_a, decoder_h, resolutions, regime):
    if index_angle == 0:
        extended_loc_orders = int(loc_orders / 2)
    else:
        extended_loc_orders = loc_orders - 1

    loc_rt_tf = tf.calculate_radii_step(D=conf.diaphragm, orders=extended_loc_orders)
    best_f0 = -float('inf')
    wait = 0

    for step in range(1, num_steps + 1):
        loc_optimizer.zero_grad()

        # --- decoding the parameters ---
        a_params = decoder_a(loc_u_a)
        h_params = decoder_h(loc_u_h)

        # 5.1. get the epsilon distribution
        # x - Tensor (Nx_resolution, 1)
        x_list = None
        if regime== (2 or 1.5):
            # h_params как вектор
            x_list = tf.material_grid_weight_2d(a_params, h_params, conf.at_hei, extended_loc_orders, conf.stack,
                                                loc_rt_tf,
                                                resolutions, Q=conf.Q, w=conf.w, LIMIT_SHAPE=conf.LIMIT_SHAPE,
                                                smooth=True, beta=beta, plot=False)
        elif regime==1:
            x_list = tf.material_grid_weight_1d(a_params, h_params, conf.at_hei, extended_loc_orders, conf.stack,
                                                loc_rt_tf,
                                                resolutions, Q=conf.Q, w=conf.w, LIMIT_SHAPE=conf.LIMIT_SHAPE,
                                                smooth=True, beta=beta, plot=False)

        # 5.2. calculate the function and its gradient using meep.adjoint.opt
        # opt expects a list of tensors x, returns a scalar f0 and a list of gradients [grad_x]
        value = opt_torch(*x_list)  # возвращает тензор-скаляр

        # 5.3. transferring the gradient to params via the chain rule:
        # dx/dparams gives us get_weights.backward(), therefore just calling .backward() from -f0 (maximize f0)
        loss = -value
        loss.backward()

        # 5.4. the gradient norm for logging
        grad_norm = 0.0

        if regime==(2 or 1.5):
            for param in (loc_u_a, loc_u_h):
                grad_norm += param.grad.detach().norm().item() ** 2
            grad_norm = grad_norm ** 0.5
        if regime==1:
            grad_norm = loc_u_a.grad.detach().norm().item()

        # 5.5. optimizer step
        loc_optimizer.step()

        # 5.6. Increasing beta to "annealing" binary masks
        beta = min(15.0, beta * 1.1)

        # 5.8. save history
        loc_history['f0'].append(value.item())
        loc_history['grad_norm'].append(grad_norm)
        loc_history['loc_a_params'].append(a_params.detach().cpu().clone())
        loc_history['loc_h_params'].append(h_params.detach().cpu().clone())
        loc_history['beta'].append(beta)

        # 5.9. logs
        if step % 1 == 0 or step == 1:
            print(
                f"Step {step:3d} | f0 = {value.item()} | beta = {beta} ||grad|| = {grad_norm} || lr = {loc_optimizer.param_groups[0]['lr']} || wait = {wait}")
        if (loc_u_a.all() or loc_u_a.all()) is None:
            print("Stopping optimization")
            break
        # early stop
        if value > best_f0 + conf.threshold:
            best_f0 = value
            wait = 0
        else:
            wait += 1
        if wait >= conf.patience:
            print(f"Early stopping at step {step}: "
                  f"f0 hasn't improved by >{conf.threshold} for {conf.threshold} steps")
            break

    # 6. At the end, params contains the optimal approximation.
    best_idx = max(range(len(loc_history['f0'])), key=lambda i: loc_history['f0'][i])
    best_a_params = loc_history['loc_a_params'][best_idx]
    best_h_params = loc_history['loc_h_params'][best_idx]
    print(f"\nBest f0 = {loc_history['f0'][best_idx]} at step {best_idx + 1}")

    return loc_history, best_a_params, best_h_params

def viz_loc_opt(
    index_angle,
    loc_history,
    best_a_params,
    best_h_params,
    loc_orders,
    is_save_params=True,
    save_path: str = None,            # <-- None = не сохранять
    save_format: str = "pdf"          # <-- "pdf" или "jpeg"
):
    """
    Visualizes the optimization process and optimized parameters.

    Parameters
    ----------
    index_angle : int
        The angle index determines the number of orders.
    loc_history : dict
        Optimization history (for example, {'f0': [...]}).
    best_a_params : torch.Tensor
        The best values of the 'a' parameter.
    best_h_params : torch.Tensor
        The best values of the 'h' parameter.
    loc_orders : int
        The number of orders.
    is_save : bool
        Whether to save the best_a and best_h parameters.
    save_path : str or None
        The path to save the final image. If None— only display.
    save_format : str
        The saving format is "pdf" or "jpeg".
    """

    # --- Appearance Settings ---
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

    # --- Number of displayed orders ---
    if index_angle == 0:
        extended_loc_orders = int(loc_orders / 2)
    else:
        extended_loc_orders = loc_orders - 1

    # --- Creating a shape with two graphs ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # --- 1. Graph of the loss function ---
    axes[0].plot(loc_history['f0'], label='f0', color='tab:blue', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Value f0')
    axes[0].set_title('Optimization progress')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- 2. Graph of optimized parameters ---
    idx = torch.arange(extended_loc_orders, dtype=torch.float32).numpy()
    axes[1].plot(idx, best_a_params, marker='o', label='a_params', color='tab:red')
    axes[1].plot(idx, best_h_params, marker='+', label='h_params', color='tab:green')
    axes[1].set_xlabel('Order')
    axes[1].set_ylabel('Parameter value')
    axes[1].set_title('Optimized parameters')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()

    # --- save and show plots ---
    if save_path:
        save_format = save_format.lower()
        if save_format not in ["pdf", "jpeg"]:
            raise ValueError("save_format must be either 'pdf' or 'jpeg'")
        fig.savefig(save_path, format=save_format, bbox_inches="tight")
        print(f"Optimization plot saved to: {save_path}")
        plt.show()
    else:
        plt.show()

    # --- save params ---
    if is_save_params:
        help_functions.save_params(params=best_a_params, name="best_a")
        help_functions.save_params(params=best_h_params, name="best_h")



def test_loc(
        index_angle: int,
        loc_orders: int,
        source: object,
        rt_np,
        resolutions,
        locNearRegion,
        best_a_params: torch.Tensor = None,
        best_h_params: torch.Tensor = None,
        is_plot_sim: bool = False,
        is_load_name_a: str = None,
        is_load_name_h: str = None,
) -> np.ndarray:
    """
    Performs a local FDTD simulation for a given angle and optimization parameters.

    If is_load=True, the parameters of the best model (best_a_params and best_h_params) are loaded.
    from files "best_a.pt " and "best_h.pt " accordingly, the arguments passed to the function are
    they are ignored.

    Parameters:
    -----------
    index_angle : int
        The index of the beam exit angle (0 ... N-1).
    loc_orders : int
        The number of local optimization orders.
    best_a_params : torch.Tensor, optional
        Optimization parameters A (optional if is_load=True).
    best_h_params : torch.Tensor, optional
        Optimization parameters H (optional if is_load=True).
    is_load : bool
        A flag indicating whether to load parameters from files instead of using the transmitted ones.

    Returns:
    --------
    None
    """
    # Loading parameters from files
    if (is_load_name_a != None) and (is_load_name_h != None):
        best_a_params = torch.load(is_load_name_a, weights_only=True)
        best_h_params = torch.load(is_load_name_h, weights_only=True)

    # Calculating additional attributes of symmetry and range of orders
    if index_angle == 0:
        extended_loc_orders = loc_orders // 2
        is_global = True
        first_index = 0
        loc_symmetry = [mp.Mirror(mp.Y)]
    else:
        extended_loc_orders = loc_orders - 1
        is_global = False
        first_index = 1
        loc_symmetry = None

    # IPlane_base, _ = nf.get_Ibase_simulation(
    #     monitor=locNearRegion,
    #     sources=source,
    #     cell=loc_cell,
    #     frequents_array=conf.inc_frequents_array,
    #     max_run_time=conf.maximum_run_time,
    #     symmetries=loc_symmetry,
    # )

    # Radius steps and design areas
    loc_rt_tf = tf.calculate_radii_step(D=conf.diaphragm, orders=extended_loc_orders)
    loc_d_rt_np, loc_mid_rt_np = nf.get_drt_midrt(rt_np, extended_loc_orders, is_global=is_global)
    loc_design_region = []

    # Generating weights on a grid
    x_test = tf.material_grid_weight_2d(
        best_a_params, best_h_params,
        conf.at_hei, extended_loc_orders,
        conf.stack, loc_rt_tf,
        resolutions,
        Q=conf.Q, w=conf.w,
        LIMIT_SHAPE=conf.LIMIT_SHAPE,
        smooth=False, beta=50, plot=False
    )

    # The basic geometry of the substrate
    geometry = [
        mp.Block(
            material=conf.in_material,
            size=mp.Vector3(conf.in_sub, mp.inf, 0),
            center=mp.Vector3(conf.sub_center.x)
        )
    ]

    # Adding design layers in local order
    for loc_order in range(first_index, extended_loc_orders):
        weight_np = x_test[loc_order].detach().cpu().numpy()
        res = resolutions[loc_order]
        material_grid = mp.MaterialGrid(
            mp.Vector3(x=res[1], y=res[0]),
            mp.air, conf.at_material,
            weights=weight_np
        )
        loc_design_region.append(
            mpa.DesignRegion(
                material_grid,
                volume=mp.Volume(
                    center=mp.Vector3(x=conf.at_hei / 2, y=loc_mid_rt_np[loc_order]),
                    size=mp.Vector3(x=conf.at_hei, y=loc_d_rt_np[loc_order] - 0.1)
                )
            )
        )
        geometry.append(
            mp.Block(
                size=mp.Vector3(x=conf.at_hei, y=loc_d_rt_np[loc_order]),
                center=mp.Vector3(x=conf.at_hei / 2 - conf.add_size / 2, y=loc_mid_rt_np[loc_order]),
                material=material_grid
            )
        )

    # Setting up the simulation
    loc_sim = mp.Simulation(
        cell_size=conf.loc_cell,
        boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.X, side=mp.ALL)],
        geometry=geometry,
        sources=source,
        default_material=mp.Medium(index=1.0),
        resolution=conf.sim_resolution,
        symmetries=loc_symmetry
    )

    # Adding near2far monitors
    near2far_regions = []
    for freq in conf.inc_frequents_array:
        near2far_regions.append(
            loc_sim.add_near2far(
                freq,   # central frequency
                conf.inc_df,  # the width of the spectrum
                1,  # discreteness
                locNearRegion
            )
        )
    #### anim-part 1
    # loc_sim.run(until=0)
    # fieldEz = np.array(loc_sim.get_array(center=mp.Vector3(0,0,0), size=conf.cell, component=mp.Ez)).T
    # fieldData = np.array([np.zeros(fieldEz.shape)])
    # def updateField(loc_sim):
    #     nonlocal fieldData
    #     fieldEz = np.array([loc_sim.get_array(center=mp.Vector3(0,0,0), size=conf.cell, component=mp.Ez)])
    #     fieldEz = np.transpose(fieldEz, (0, 2, 1))
    #     fieldData = np.vstack((fieldData, fieldEz))
    #
    # def animate(i):
    #     im.set_data(fieldData[i])
    #     return im,
    ####

    # Running the simulation
    loc_sim.run(until=conf.maximum_run_time)

    #### anim-part 2
    # animationTimestepDuration = 0.7
    # endTime = conf.maximum_run_time + 30
    # loc_sim.run(mp.at_every(animationTimestepDuration, updateField), until=endTime)
    # fieldData = np.real(fieldData)
    ####

    # Visualization of the field
    if is_plot_sim:
        plt.figure(figsize=(2, 50))
        loc_sim.plot2D(
            fields=mp.Ez,
            output_plane=mp.Volume(center=mp.Vector3(0, 0), size=conf.loc_cell)
        )
        plt.show()

    IOptimized = nf.get_I_loc(loc_sim, near2far_regions, 0, conf.ff_res, mp.Vector3(0, conf.spot_ff, 0), f=conf.f)

    #### anim-part 3
    # abs_max = np.abs(fieldData).max()/2
    # print("Максимальная амплитуда поля:", abs_max)
    #
    # fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    # im = ax.imshow(fieldData[0], cmap='RdBu',
    #                vmin=-abs_max, vmax=abs_max,
    #                extent=[-conf.cell.x/2, +conf.cell.x/2, -conf.cell.y/2, +conf.cell.y/2],
    #                origin='lower')
    # fig.colorbar(im)
    # numberAnimationTimesteps = fieldData.shape[0]
    # fieldAnimation = animation.FuncAnimation(fig, animate, frames=numberAnimationTimesteps, interval=20, blit=True)
    # fieldAnimation.save('foucusing_animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
    # plt.show()
    ####

    return IOptimized


def update_global_params(
        index_angle: int,
        global_a_params: torch.Tensor,
        global_h_params: torch.Tensor,
        loc_best_a_params: torch.Tensor,
        loc_best_h_params: torch.Tensor,
        is_save: bool
) -> None:
    """
    Overwrites the range of elements in global tensors torch.float64
    local best tensors, adjusting the range to the length of the local data.

    Parameters:
    -----------
    global_a_params : torch.Tensor, shape (N,), dtype=torch.float64
    global_h_params : torch.Tensor, shape (N,), dtype=torch.float64
    loc_best_a_params : torch.Tensor, shape (M,), dtype=torch.float64
    loc_best_h_params : torch.Tensor, shape (M,), dtype=torch.float64
    index_angle : Any
        The argument for nf.get_indx_orders, which returns (indx_start, indx_end).
    nf : object
        Using the get_indx_orders(angle) → method Tuple[int, int].
    """

    # 1. get the initial and final indexes
    indx_start, indx_end = nf.get_indx_orders(index_angle)
    indx_start = int(indx_start)
    indx_end = int(indx_end)
    actual_len = loc_best_a_params.numel()

    # 2. adjusting the end of the range to the difference between the local and current length
    diff = actual_len - (indx_end - indx_start)
    indx_end += diff

    # 3. crop along the boundaries of global tensors
    max_len = min(global_a_params.numel(), global_h_params.numel())
    indx_start = max(0, indx_start)
    indx_end = min(indx_end, max_len)

    # 4. calculating the actual number of copied elements
    length_to_copy = min(indx_end - indx_start, actual_len)
    if length_to_copy <= 0:
        return

    # 5. batch copying
    global_a_params[indx_start:indx_start + length_to_copy] = loc_best_a_params[:length_to_copy]
    global_h_params[indx_start:indx_start + length_to_copy] = loc_best_h_params[:length_to_copy]

    if is_save:
        help_functions.save_params(global_a_params, name=f"global_a_params_{len(global_a_params)}")
        help_functions.save_params(global_h_params, name=f"global_h_params_{len(global_h_params)}")


def get_local_tensor_from_global(global_params, index_angle, loc_orders):
    """
    Extracts a plot from global_params in the range of indexes defined by index_angle.
    If the final index goes beyond the tensor, it adds 1.0 values to the tensor and outputs a warning.

    Args:
        global_params (torch.Tensor): A global parameter tensor.
        index_angle: The angle defining the start and end indexes.

    Returns:
        torch.Tensor: The extracted tensor.
    """

    len_global = len(global_params)

    if index_angle == 0:
        indx_start, indx_end = nf.get_indx_orders(index_angle)
    else:
        indx_start, _ = nf.get_indx_orders(index_angle)
        indx_end = indx_start + loc_orders

    # If there are enough elements, extract the desired area.
    if indx_end <= len_global:
        local_tensor = global_params[indx_start:indx_end]
    else:
        # Extracting all available elements
        available = len_global - indx_start
        local_tensor = global_params[indx_start:len_global]
        # Creating a complement
        padding = torch.ones(indx_end - len_global, dtype=global_params.dtype)
        # Combining the extracted elements with the addition
        local_tensor = torch.cat((local_tensor, padding))
        print(f"Last index ({indx_end}) over {len_global} of global index."
              f" Added {indx_end - available} ones indexes")

    return local_tensor