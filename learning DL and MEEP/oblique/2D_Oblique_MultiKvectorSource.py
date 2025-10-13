import meep as mp
import numpy as np
import conf
import matplotlib.pyplot as plt
from lib import my_sources
from matplotlib import animation

# Simulation parameters for MEEP
increase_resolution = 10 # by default, that should be 1. This parameter multiply min_resolution
cell = mp.Vector3(0.02*1e3, 0.025*1e3, 0)  # размер области симуляции
pml_layers = [mp.PML(thickness=conf.pml, direction=mp.X, side=mp.ALL)]

sources_obj = my_sources.MultiKvectorSource(y=cell.y, frequency=conf.inc_fcen, dfreq= conf.inc_df, angles_deg = conf.angle_list, center_y= cell.y/2)
sources = sources_obj.sources

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    sources=sources,
    resolution=conf.min_resolution * increase_resolution,

)

sim.run(until=0)
fieldEz = np.array(sim.get_array(center=mp.Vector3(0,0,0), size=cell, component=mp.Ez)).T
fieldData = np.array([np.zeros(fieldEz.shape)])
def updateField(sim):
    global fieldData
    fieldEz = np.array([sim.get_array(center=mp.Vector3(0,0,0), size=cell, component=mp.Ez)])
    fieldEz = np.transpose(fieldEz, (0, 2, 1))
    fieldData = np.vstack((fieldData, fieldEz))

def animate(i):
    im.set_data(fieldData[i])
    return im,

geometry = []

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=conf.min_resolution * increase_resolution,

)

plt.figure(figsize=(5, 10))
sim.plot2D(
    output_plane = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(cell[0], cell[1],0))
)
plt.show()  # Отображение изображения

animationTimestepDuration = 0.1
endTime = 20

# Running the simulation
sim.run(mp.at_every(animationTimestepDuration, updateField), until=endTime)
fieldData = np.real(fieldData)

fig, ax = plt.subplots()
im = ax.imshow(fieldData[0], cmap='RdBu', vmin=-1e-5, vmax=1e-5, extent=[-cell.x/2, +cell.x/2, -cell.y/2, cell.y/2], origin='lower')
fig.colorbar(im)
numberAnimationTimesteps = fieldData.shape[0]
fieldAnimation = animation.FuncAnimation(fig, animate, frames=numberAnimationTimesteps, interval=20, blit=True)
fieldAnimation.save('ang45_KvectorSource_animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
plt.show()