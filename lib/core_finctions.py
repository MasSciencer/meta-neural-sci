import numpy as np
import meep as mp
import torch
from configs import conf_test as conf
from lib import adjoint as mpa
from lib import torch_functions as tf
from lib import numpy_functions as nf
from lib import help_functions
from matplotlib import pyplot as plt
from matplotlib import animation


def init_opt(
        index_angle: int,
        obj_func,
        a_initial,
        h_initial,
        rt_np, drt_np, midrt_np,
        resolutions,
        stack_batch_src,
        NearRegion,
        is_center,
        symmetry,
        is_plot=False,
        is_plot_cross=False,
        is_reflection=False,
):
    """
    The function of starting a local cell
    """

    # pure way
    refl_I_pure = None
    if is_reflection:
        geometry = [mp.Block(
            material=conf.in_material,
            size=mp.Vector3(conf.in_sub, mp.inf, 0),
            center=mp.Vector3(conf.sub_center.x),
        ), mp.Block(
            material=conf.in_material_buffer,
            size=mp.Vector3(conf.in_sub_buffer, mp.inf, 0),
            center=mp.Vector3(conf.sub_center_buffer.x),
        )]

        loc_sim = mp.Simulation(
            cell_size=conf.loc_cell,
            boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.X)],
            geometry=geometry,
            sources=stack_batch_src[index_angle],
            default_material=mp.Medium(index=1.0),
            resolution=conf.sim_resolution,
            symmetries=symmetry,
        )
        refl_flux = loc_sim.add_near2far(
            conf.inc_fcen,  # central frequency
            conf.inc_df,    # the width of the spectrum
            1,              # discreteness
            NearRegion
        )
        loc_sim.run(until=conf.maximum_run_time)
        # refl_dtf_pure = loc_sim.get_near2far_data(refl_flux)
        refl_I_pure = nf.get_I(
            loc_sim,
            refl_flux,
            conf.ff_res,
            mp.Vector3(0, conf.spot_ff, 0),
            out_angle_in_deg=0,
            f=conf.f
        )

    # main way
    loc_design_region = []
    loc_focus_point_list = []
    point1 = mp.Vector3(x=conf.f, y=0)
    loc_focus_point_list.append(point1)

    rt_tf = torch.from_numpy(rt_np).float()
    drt_tf = torch.from_numpy(drt_np).float()
    list_loc_x0 = tf.material_grid_weight_2d(a_initial, h_initial, conf.at_hei, conf.stack,
                                             rt_tf = rt_tf,  drt_tf = drt_tf, resolutions = resolutions, Q=conf.Q, w=conf.w,
                                             LIMIT_SHAPE=conf.LIMIT_SHAPE, smooth=False, beta=50.0, plot=is_plot_cross)
    geometry = [mp.Block(
        material=conf.in_material,
        size=mp.Vector3(conf.in_sub, mp.inf, 0),
        center=mp.Vector3(conf.sub_center.x),
    )]

    if is_reflection:
        geometry = [mp.Block(
            material=conf.in_material,
            size=mp.Vector3(conf.in_sub, mp.inf, 0),
            center=mp.Vector3(conf.sub_center.x),
        ), mp.Block(
            material=conf.in_material_buffer,
            size=mp.Vector3(conf.in_sub_buffer, mp.inf, 0),
            center=mp.Vector3(conf.sub_center_buffer.x),
        )]

    for order in range(0, len(a_initial)):
        weight_numpy = list_loc_x0[order].detach().cpu().numpy()
        res = resolutions[order]
        material_grid = mp.MaterialGrid(
            mp.Vector3(x=res[1], y=res[0]),
            mp.air,
            conf.at_material,
            weights=weight_numpy
        )

        if is_center:
            loc_design_region.append(mpa.DesignRegion(
                material_grid,
                volume=mp.Volume(center=mp.Vector3(x=conf.at_center.x, y=midrt_np[order]),
                                 size=mp.Vector3(x=conf.at_hei, y=drt_np[order]))
            ))
            geometry.append(mp.Block(
                size=mp.Vector3(x=conf.at_hei, y=drt_np[order]),
                center=mp.Vector3(x=conf.at_center.x, y=midrt_np[order]),
                material=material_grid,
            ))
            if order > 0:
                geometry.append(mp.Block(
                    size=mp.Vector3(x=conf.at_hei, y=drt_np[order]),
                    center=mp.Vector3(x=conf.at_center.x, y=-midrt_np[order]),
                    material=material_grid,
                ))
        else:
            loc_design_region.append(mpa.DesignRegion(
                material_grid,
                volume=mp.Volume(center=mp.Vector3(x=conf.at_center.x, y=midrt_np[order]),
                                 size=mp.Vector3(x=conf.at_hei, y=drt_np[order]))
            ))
            geometry.append(mp.Block(
                size=mp.Vector3(x=conf.at_hei, y=drt_np[order]),
                center=mp.Vector3(x=conf.at_center.x, y=midrt_np[order]),
                material=material_grid,
            ))

    loc_sim = mp.Simulation(
        cell_size=conf.loc_cell,
        boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.X)],
        geometry=geometry,
        sources=stack_batch_src[index_angle],
        default_material=mp.Medium(index=1.0),
        resolution=conf.sim_resolution,
        symmetries=symmetry,
    )

    # FarFields = mpa.Near2FarFields(loc_sim, [locNearRegion], loc_focus_point_list, norm_near_fields=refl_dtf_pure)
    FarFields = mpa.Near2FarFields(loc_sim, [NearRegion], loc_focus_point_list)
    ob_list = [FarFields]

    loc_opt = mpa.OptimizationProblem(
        simulation=loc_sim,
        objective_functions=[obj_func],
        objective_arguments=ob_list,
        design_regions=loc_design_region,
        frequencies=conf.inc_frequents_array,
        maximum_run_time=conf.maximum_run_time,
        until_steps=conf.maximum_run_time,
    )

    if is_plot:
        plt.figure(figsize=(4, 50))
        loc_opt.plot2D(True)

    if is_reflection:
        return loc_opt, refl_I_pure
    else:
        return loc_opt

def init_loc_opt_rotate(
        index_angle: int,
        loc_orders: int,
        obj_func,
        a_initial,
        h_initial,
        rt_np,
        resolutions,
        stack_batch_src,
        locNearRegion,
        is_plot=False,
        is_plot_cross=False,
):
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

    loc_d_rt_np, loc_mid_rt_np = nf.get_drt_midrt(rt_np, extended_loc_orders, is_global=is_global)
    loc_rt_tf = tf.calculate_radii_step(D=conf.diaphragm, orders=extended_loc_orders)

    # main way
    loc_design_region = []
    loc_focus_point_list = []
    point1 = mp.Vector3(x=conf.f, y=0)
    loc_focus_point_list.append(point1)
    list_loc_x0 = tf.material_grid_weight_2d_rotate(a_initial, h_initial, conf.at_hei, conf.stack,
                                             loc_rt_tf, resolutions, Q=conf.Q, w=conf.w,
                                             LIMIT_SHAPE=conf.LIMIT_SHAPE, smooth=False, beta=50.0, plot=is_plot_cross)
    geometry = [mp.Block(
        material=conf.in_material,
        size=mp.Vector3(conf.in_sub, mp.inf, 0),
        center=mp.Vector3(conf.sub_center.x),
    )]

    for loc_order in range(first_index, extended_loc_orders):
        weight_numpy = list_loc_x0[
            loc_order].detach().cpu().numpy()
        res = resolutions[loc_order]
        material_grid = mp.MaterialGrid(
            mp.Vector3(x=res[1], y=res[0]),
            mp.air,
            conf.at_material,
            weights=weight_numpy
        )
        loc_design_region.append(mpa.DesignRegion(
            material_grid,
            volume=mp.Volume(center=mp.Vector3(x=conf.at_center.x, y=loc_mid_rt_np[loc_order]),
                             size=mp.Vector3(x=conf.at_hei, y=loc_d_rt_np[loc_order] - 0.1)) # 0.1 - offset from the edge of the geometry so that the orders do not run into each other
        ))

        geometry.append(mp.Block(
            size=mp.Vector3(x=conf.at_hei, y=loc_d_rt_np[loc_order]),
            center=mp.Vector3(x=conf.at_center.x, y=loc_mid_rt_np[loc_order]),
            material=material_grid,
        ))

    loc_sim = mp.Simulation(
        cell_size=conf.loc_cell,
        boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.X)],
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
        objective_arguments=ob_list,
        design_regions=loc_design_region,
        frequencies=conf.inc_frequents_array,
        maximum_run_time=conf.maximum_run_time,
        until_steps=conf.maximum_run_time,
    )

    if is_plot:
        plt.figure(figsize=(2, 50))
        loc_opt.plot2D(True)

    return loc_opt

def init_loc_opt_hyper(a_initial, h_initial, p_min, p_max, h_min, h_max, lr, regime):
    def encode_param(value, vmin, vmax):
        # нормализуем значение в [0,1] и применяем logit для перевода в (-inf, inf)
        norm = (value - vmin) / (vmax - vmin)
        # защитимся от выхода за диапазон
        norm = torch.clamp(norm, 1e-6, 1 - 1e-6)
        return torch.logit(norm)

    loc_optimizer = None
    loc_u_a = None
    loc_u_h = None
    scheduler = None

    # --- a_initial ---
    if isinstance(a_initial, torch.Tensor):
        loc_a_params = a_initial
    else:
        raise TypeError("a_initial must be torch.Tensor")

    # --- h_initial ---
    if isinstance(h_initial, torch.Tensor):
        loc_h_params = h_initial
    else:
        raise TypeError("h_initial must be torch.Tensor")

    # --- logit ---
    loc_u_a0 = encode_param(loc_a_params, p_min, p_max)

    if regime==2:
        loc_u_h0 = encode_param(loc_h_params, h_min, h_max)
        loc_u_a = torch.nn.Parameter(loc_u_a0)
        loc_u_h = torch.nn.Parameter(loc_u_h0)

        loc_optimizer = torch.optim.Adam(
            [loc_u_a, loc_u_h],
            lr=lr,
            betas=(0.8, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            loc_optimizer,
            T_0=5,
            T_mult=1,
            eta_min=lr * 0.5
        )

    elif regime == 1.5:
        if isinstance(h_initial, torch.Tensor):
            scalar_h0 = h_initial.flatten()[0]
        else:
            scalar_h0 = torch.tensor(h_initial, dtype=torch.float64)
        loc_u_h0 = encode_param(scalar_h0, h_min, h_max)
        loc_u_a = torch.nn.Parameter(loc_u_a0)
        init_u_h = torch.nn.Parameter(loc_u_h0)
        loc_u_h = init_u_h.expand(loc_h_params.shape[0])
        loc_optimizer = torch.optim.Adam(
            [loc_u_a, init_u_h],
            lr=lr,
            betas=(0.8, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            loc_optimizer,
            T_0=5,
            T_mult=1,
            eta_min=lr * 0.5
    )

    elif regime==1:
        loc_u_h0 = encode_param(loc_h_params, h_min, h_max)
        loc_u_a = torch.nn.Parameter(loc_u_a0)
        loc_u_h = loc_u_h0
        loc_optimizer = torch.optim.Adam(
            [loc_u_a],
            lr=lr,
            betas=(0.8, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            loc_optimizer,
            T_0=5,
            T_mult=1,
            eta_min=lr * 0.5
        )

    else:
        ValueError("regime must be 1, 2 or 1.5")

    # History
    store_loc_history = {
        'f0': [],
        'grad_norm': [],
        'loc_a_params': [],
        'loc_h_params': [],
        'beta': []
    }

    return loc_optimizer, scheduler, store_loc_history, loc_u_a, loc_u_h


def run_loc_opt(opt_torch, loc_rt_np, loc_drt_np, loc_optimizer, scheduler, loc_u_a, loc_u_h, loc_history, beta, num_steps, decoder_a, decoder_h, resolutions, regime):
    best_f0 = -float('inf')
    wait = 0
    rt_tf = torch.from_numpy(loc_rt_np).float()
    drt_tf = torch.from_numpy(loc_drt_np).float()

    for step in range(1, num_steps + 1):
        loc_optimizer.zero_grad()

        # --- decoding the parameters ---
        a_params = decoder_a(loc_u_a)
        h_params = decoder_h(loc_u_h)

        # 5.1. get the epsilon distribution
        # x - Tensor (Nx_resolution, 1)
        x_list = None
        if regime == 2 or regime == 1.5:
            # h_params как вектор
            x_list = tf.material_grid_weight_2d(a_params, h_params, conf.at_hei, conf.stack,
                                                rt_tf=rt_tf, drt_tf=drt_tf,
                                                resolutions=resolutions, Q=conf.Q, w=conf.w, LIMIT_SHAPE=conf.LIMIT_SHAPE,
                                                smooth=True, beta=beta, plot=False)
        elif regime==1:
            x_list = tf.material_grid_weight_1d(a_params, h_params, conf.at_hei, conf.stack,
                                                rt_tf=rt_tf, drt_tf=drt_tf,
                                                resolutions=resolutions, Q=conf.Q, w=conf.w, LIMIT_SHAPE=conf.LIMIT_SHAPE,
                                                smooth=True, beta=beta, plot=False)

        # 5.2. calculate the function and its gradient using meep.adjoint.opt
        # opt expects a list of tensors x, returns a scalar f0 and a list of gradients [grad_x]
        value = opt_torch(*x_list)  # возвращает тензор-скаляр

        # 5.3. transferring the gradient to params via the chain rule:
        # dx/dparams gives us get_weights.backward(), therefore just calling .backward() from -f0 (maximize f0)
        loss = -value
        loss.backward()

        # 5.4. the gradient norm for logging
        grad_norm_sq = 0.0
        for param in (loc_u_a, loc_u_h):
            if param is None or param.grad is None:
                continue
            g = param.grad.detach()
            grad_norm_sq += g.norm().item() ** 2
        grad_norm = grad_norm_sq ** 0.5

        # 5.5. optimizer step
        loc_optimizer.step()
        scheduler.step()
        mean_eff_lr = tf.get_effective_lr(loc_optimizer)

        # 5.6. Increasing beta to "annealing" binary masks
        beta = min(50.0, beta * 1.07)

        # 5.8. save history
        loc_history['f0'].append(value.item())
        loc_history['grad_norm'].append(grad_norm)
        loc_history['loc_a_params'].append(a_params.detach().cpu().clone())
        loc_history['loc_h_params'].append(h_params.detach().cpu().clone())
        loc_history['beta'].append(beta)

        # 5.9. logs
        if step % 1 == 0 or step == 1:
            print(f"Step {step:2d} ({num_steps + 1}) | f0 = {value.item()} | beta = {beta} ||grad|| = {grad_norm} || wait = {wait}"),
            print(f"Mean effective LR = {mean_eff_lr:.3e}"),
            print(f"lr = {scheduler.get_last_lr()[0]:.3f}")

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
                  f"f0 hasn't improved by >{conf.threshold} for {conf.patience} steps")
            break

    # 6. At the end, params contains the optimal approximation.
    best_idx = max(range(len(loc_history['f0'])), key=lambda i: loc_history['f0'][i])
    best_a_params = loc_history['loc_a_params'][best_idx]
    best_h_params = loc_history['loc_h_params'][best_idx]
    print(f"\nBest f0 = {loc_history['f0'][best_idx]} at step {best_idx + 1}")

    return loc_history, best_a_params, best_h_params

def viz_loc_opt(
    loc_history,
    best_a_params,
    best_h_params,
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
    idx = torch.arange(len(best_a_params), dtype=torch.float32).numpy()
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


def test_loc(
        source: object,
        rt_np, drt_np, midrt_np,
        resolutions,
        is_center,
        loc_symmetry,
        locNearRegion,
        best_a_params: torch.Tensor = None,
        best_h_params: torch.Tensor = None,
        cut_shift: float = 0.0,
        is_reflection=False,
        is_plot_sim: bool = False,
        is_plot_cross: bool = False

) -> np.ndarray:
    """
    Performs a local FDTD simulation for a given angle and optimization parameters.

    Parameters:
    -----------
    index_angle : int
        The index of the beam exit angle (0 ... N-1).
    loc_orders : int
        The number of local optimization orders.
    best_a_params : torch.Tensor, optional
        Optimization parameters A
    best_h_params : torch.Tensor, optional
        Optimization parameters H

    Returns:
    --------
    None
    """
    rt_tf = torch.from_numpy(rt_np).float()
    drt_tf = torch.from_numpy(drt_np).float()

    # Generating weights on a grid
    x_test = tf.material_grid_weight_2d(
        best_a_params, best_h_params,
        conf.at_hei,
        conf.stack,
        rt_tf = rt_tf, drt_tf = drt_tf,
        resolutions = resolutions,
        cut_shift = cut_shift,
        Q=conf.Q, w=conf.w,
        LIMIT_SHAPE=conf.LIMIT_SHAPE,
        smooth=False, beta=50, plot=is_plot_cross
    )

    # The basic geometry of the substrate
    geometry = [
        mp.Block(
            material=conf.in_material,
            size=mp.Vector3(conf.in_sub, mp.inf, 0),
            center=mp.Vector3(conf.sub_center.x)
        )
    ]

    if is_reflection:
        geometry = [mp.Block(
            material=conf.in_material,
            size=mp.Vector3(conf.in_sub, mp.inf, 0),
            center=mp.Vector3(conf.sub_center.x),
        ), mp.Block(
            material=conf.in_material_buffer,
            size=mp.Vector3(conf.in_sub_buffer, mp.inf, 0),
            center=mp.Vector3(conf.sub_center_buffer.x),
        )]



    for order in range(0, len(best_a_params)):
        weight_numpy = x_test[order].detach().cpu().numpy()
        res = resolutions[order]
        material_grid = mp.MaterialGrid(
            mp.Vector3(x=res[1], y=res[0]),
            mp.air,
            conf.at_material,
            weights=weight_numpy
        )

        if is_center:
            geometry.append(mp.Block(
                size=mp.Vector3(x=conf.at_hei, y=drt_np[order]),
                center=mp.Vector3(x=conf.at_center.x, y=midrt_np[order]),
                material=material_grid,
            ))
            if order > 0:
                geometry.append(mp.Block(
                    size=mp.Vector3(x=conf.at_hei, y=drt_np[order]),
                    center=mp.Vector3(x=conf.at_center.x, y=-midrt_np[order]),
                    material=material_grid,
                ))
        else:
            geometry.append(mp.Block(
                size=mp.Vector3(x=conf.at_hei, y=drt_np[order]),
                center=mp.Vector3(x=conf.at_center.x, y=midrt_np[order]),
                material=material_grid,
            ))

    # Setting up the simulation
    loc_sim = mp.Simulation(
        cell_size=conf.loc_cell,
        boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.X)],
        geometry=geometry,
        sources=source,
        default_material=mp.Medium(index=1.0),
        resolution=conf.sim_resolution,
        symmetries=loc_symmetry
    )

    if is_plot_sim:
        plt.figure(figsize=(2, 50))
        loc_sim.plot2D()

    # Adding near2far monitors
    near2far_regions = []
    for i in range (len(conf.inc_frequents_array)):
        one_freq_n2f =  loc_sim.add_near2far(
            conf.inc_frequents_array[i],   # central frequency
            conf.inc_df,                   # the width of the spectrum
            1,                             # discreteness
            locNearRegion
        )
        # if minusFF is not None:
        #     loc_sim.load_near2far_data(one_freq_n2f, minusFF)
        near2far_regions.append(one_freq_n2f)

    #### anim-part 1
    # loc_sim.run(until=0)
    # fieldEz = np.array(loc_sim.get_array(center=mp.Vector3(0,0,0), size=conf.loc_cell, component=mp.Ez)).T
    # fieldData = np.array([np.zeros(fieldEz.shape)])
    # def updateField(loc_sim):
    #     nonlocal fieldData
    #     fieldEz = np.array([loc_sim.get_array(center=mp.Vector3(0,0,0), size=conf.loc_cell, component=mp.Ez)])
    #     fieldEz = np.transpose(fieldEz, (0, 2, 1))
    #     fieldData = np.vstack((fieldData, fieldEz))
    #
    # def animate(i):
    #     im.set_data(fieldData[i])
    #     return im,
    ####

    # Running the simulation !!! TURN OFF FOR ANIMATION
    loc_sim.run(until=conf.maximum_run_time)

    #### anim-part 2
    # animationTimestepDuration = 1.0
    # endTime = conf.maximum_run_time
    # loc_sim.run(mp.at_every(animationTimestepDuration, updateField), until=endTime)
    # fieldData = np.real(fieldData)
    ####

    # # Visualization of the field
    # if is_plot_sim:
    #     plt.figure(figsize=(2, 50))
    #     loc_sim.plot2D(
    #         fields=mp.Ez,
    #         output_plane=mp.Volume(center=mp.Vector3(0, 0), size=conf.loc_cell)
    #     )
    #     plt.show()

    res_plane_I = nf.get_I_loc(
        sim = loc_sim,
        near2far_region = near2far_regions,
        shift = 0,
        ff_res = conf.ff_res,
        size = mp.Vector3(0, conf.spot_ff, 0),
        f=conf.f
    )

    res_deep_I = nf.get_I_loc(
        sim = loc_sim,
        near2far_region=near2far_regions,
        shift = 0,
        ff_res=conf.ff_res,
        size=mp.Vector3(conf.spot_ff, 0, 0),
        f=conf.f
    )

    #### anim-part 3
    # abs_max = np.abs(fieldData).max()/2
    # fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    # im = ax.imshow(fieldData[0], cmap='RdBu',
    #                vmin=-abs_max, vmax=abs_max,
    #                extent=[-conf.loc_cell.x/2, +conf.loc_cell.x/2, -conf.loc_cell.y/2, +conf.loc_cell.y/2],
    #                origin='lower')
    # fig.colorbar(im)
    # numberAnimationTimesteps = fieldData.shape[0]
    # fieldAnimation = animation.FuncAnimation(fig, animate, frames=numberAnimationTimesteps, interval=40, blit=True)
    # fieldAnimation.save('foucusing_animation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
    # plt.show()
    ####

    return res_plane_I, res_deep_I

def test_loc_rotate(
        index_angle: int,
        loc_orders: int,
        source: object,
        rt_np,
        resolutions,
        locNearRegion,
        best_a_params: torch.Tensor = None,
        best_h_params: torch.Tensor = None,
        cut_shift: float = 0.0,
        is_plot_sim: bool = False,
        is_plot_cross: bool = False
) -> np.ndarray:
    """
    Performs a local FDTD simulation for a given angle and optimization parameters.

    Parameters:
    -----------
    index_angle : int
        The index of the beam exit angle (0 ... N-1).
    loc_orders : int
        The number of local optimization orders.
    best_a_params : torch.Tensor, optional
        Optimization parameters A
    best_h_params : torch.Tensor, optional
        Optimization parameters H

    Returns:
    --------
    None
    """
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

    # Radius steps and design areas
    loc_rt_tf = tf.calculate_radii_step(D=conf.diaphragm, orders=extended_loc_orders)
    loc_d_rt_np, loc_mid_rt_np = nf.get_drt_midrt(rt_np, extended_loc_orders, is_global=is_global)
    loc_design_region = []

    # Generating weights on a grid
    x_test = tf.material_grid_weight_2d_rotate(
        best_a_params, best_h_params,
        conf.at_hei,
        conf.stack,
        loc_rt_tf,
        resolutions,
        cut_shift = cut_shift,
        Q=conf.Q, w=conf.w,
        LIMIT_SHAPE=conf.LIMIT_SHAPE,
        smooth=False, beta=50, plot=is_plot_cross
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
        weight_np = x_test[
            loc_order].detach().cpu().numpy()
        res = resolutions[loc_order]
        material_grid = mp.MaterialGrid(
            mp.Vector3(x=res[1], y=res[0]),
            mp.air,
            conf.at_material,
            weights=weight_np
        )
        loc_design_region.append(mpa.DesignRegion(
            material_grid,
            volume=mp.Volume(center=mp.Vector3(x=conf.at_center.x, y=loc_mid_rt_np[loc_order]),
                             size=mp.Vector3(x=conf.at_hei, y=loc_d_rt_np[loc_order] - 0.1)) # 0.1 - offset from the edge of the geometry so that the orders do not run into each other
        ))

        geometry.append(mp.Block(
            size=mp.Vector3(x=conf.at_hei, y=loc_d_rt_np[loc_order]),
            center=mp.Vector3(x=conf.at_center.x, y=loc_mid_rt_np[loc_order]),
            material=material_grid,
        ))

    # Setting up the simulation
    loc_sim = mp.Simulation(
        cell_size=conf.loc_cell,
        boundary_layers=[mp.PML(thickness=conf.pml, direction=mp.X)],
        geometry=geometry,
        sources=source,
        default_material=mp.Medium(index=1.0),
        resolution=conf.sim_resolution,
        symmetries=loc_symmetry
    )

    # Adding near2far monitors
    near2far_regions = []
    for i in range (len(conf.inc_frequents_array)):
        one_freq_n2f =  loc_sim.add_near2far(
            conf.inc_frequents_array[i],   # central frequency
            conf.inc_df,                   # the width of the spectrum
            1,                             # discreteness
            locNearRegion
        )
        near2far_regions.append(one_freq_n2f)

    # Running the simulation !!! TURN OFF FOR ANIMATION
    loc_sim.run(until=conf.maximum_run_time)

    # Visualization of the field
    if is_plot_sim:
        plt.figure(figsize=(2, 50))
        loc_sim.plot2D(
            fields=mp.Ez,
            output_plane=mp.Volume(center=mp.Vector3(0, 0), size=conf.loc_cell)
        )
        plt.show()

    res_plane_I = nf.get_I_loc(
        sim = loc_sim,
        near2far_region = near2far_regions,
        shift = 0,
        ff_res = conf.ff_res,
        size = mp.Vector3(0, conf.spot_ff, 0),
        f=conf.f
    )

    res_deep_I = nf.get_I_loc(
        sim = loc_sim,
        near2far_region=near2far_regions,
        shift=0,
        ff_res=conf.ff_res,
        size=mp.Vector3(conf.spot_ff, 0, 0),
        f=conf.f
    )

    return res_plane_I, res_deep_I


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
        indx_end = indx_start + loc_orders-1

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
