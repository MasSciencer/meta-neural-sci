# Настройка проекта
import meep as mp
import conf
import torch
from lib import torch_functions as tf
from lib import numpy_functions as nf
from lib import my_gui, my_decoders, my_sources
import numpy as np
mp.verbosity(0) # decrease amount of output

"""
Base parameters
"""
# 1. Shape
global_index_angle = None
rt_tf = tf.calculate_radii_step(D=conf.aperture, orders=conf.orders)
rt_np = nf.calculate_radii_step(D=conf.aperture, orders=conf.orders)
d_rt_np, mid_rt_np = nf.get_drt_midrt(rt_np, conf.orders, is_global = True)

# 1.1 Cписок разрешений
gen_resolution = nf.AdaptiveResolution(
    drt=d_rt_np,
    x_dim=conf.x_dim,
    h_dim= conf.h_dim,
)
resolutions_objs = gen_resolution.generate()
resolutions_tuples = [(r.w, r.h) for r in resolutions_objs]
sim_resolution = conf.min_resolution * conf.coef_resolution

angle_in_rad = np.radians(conf.angle_list)
angle_in_deg = conf.angle_list
angle_after_refraction_in_radii = my_sources.refraction_input(angle_in_deg, conf.in_epsilon)
angle_after_refraction_in_deg = np.rad2deg(angle_after_refraction_in_radii)

global_a_params = torch.load("global_a_params_20.0.pt", weights_only=True)
global_h_params = torch.load("global_h_params_20.0.pt", weights_only=True)
nf.generate_lens_stl_linear(global_a_params, global_h_params, conf.orders, rt_np, Q = conf.Q, w = conf.w, filename='gds_2param.stl')
