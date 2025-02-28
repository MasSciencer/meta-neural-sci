import matplotlib.pyplot as plt
import numpy as np
import meep as mp
from IPython.display import clear_output
import conf
import math

# Параметры симуляции
pml_layers = [mp.PML(thickness=2)]
resolution = 20
wavelength = 1.0  # длина волны в мкм

# Создание структуры с мета-линзой
geometry = [mp.Cylinder(radius=1.0, height=1.0, material=mp.Medium(index=3))]

# Установка угла падения волны
theta = math.radians(45)  # Угол в радианах (45 градусов)
k_point = mp.Vector3(math.cos(theta), math.sin(theta)).scale(2 * math.pi / wavelength)

# Определяем источник
source = [mp.Source(
    mp.ContinuousSource(conf.inc_fcen, fwidth=conf.inc_df, is_integrated=True),
    component=mp.Ez,
    center=mp.Vector3(x=-5, y=0, z=0),
    size = mp.Vector3(x=0, y=5, z=0),
)]

# Установка симуляции
sim = mp.Simulation(
    cell_size=mp.Vector3(20, 20, 0),  # Размер ячейки
    geometry=geometry,
    sources=source,
    k_point=k_point,
    resolution=resolution,
    boundary_layers=pml_layers
)

sim.run(until=100)

plt.figure(dpi=100)
sim.plot2D(fields=mp.Ez)
plt.show()