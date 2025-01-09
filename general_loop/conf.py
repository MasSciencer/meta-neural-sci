""""
Notice:
- all units in microns
- by default, any place where there are no objects there is air (ε=1)
 although this can be changed by setting the default_material variable.
-
-
"""

import meep as mp
from meep.materials import aSi

# 1. Settings for defining meta-atom parameters. Describe the shape of your meta-atom.
# this is about PML
in_pml = 2.0  # Default = 1 micron. PML thickness （Определяет толщину абсолютно поглощающего слоя, который ограничивает зону вычисления)
in_sub = 1.0  # Depends on a wavelength = 2 microns. substrate thickness (Толщина подложки)
in_gap = 1.0  # Depends on idk = 1 microns. padding between grating and PML (Расстояние между мета-атомом и полностью поглощающим слоем)
in_material = mp.Medium(epsilon=11.56) # n≈sqrl(epsilon) for a wavelength of 3.7 microns, the refractive index is 3.4 for amorphous silicon

# this is about meta-atom
at_per = 10  # 8 microns. grating periodicity (Частота повторения ячейки). must be larger than meta-atom
at_material = mp.Medium(epsilon=11.56) # Для Si. https://refractiveindex.info/?shelf=main&book=Si&page=Li-293K. Потом заменю на библиотеку материалов.
at_hei = 2  # meta-atom height (equals Y axis in global axis)
at_W = 4 # long of meta-atom (equals X axis in global axis)
at_H = 3 # width of meta-atom (equals Z axis in global axis)
#depended
cell_X = in_pml + in_sub + at_hei + in_gap + in_pml # Size of whole simulation a cell.
cell_Y = at_per # Size of compute a cell. The Y size of cell simulation
cell_Z = at_H + in_gap # Size on compute a cell. The Z size of cell
sub_center = mp.Vector3(x=- 0.5 * at_hei, y=0, z=cell_Z/2)
at_center = mp.Vector3(0.5 * in_sub)

# 2. Settings for determining the radiation source
inc_lcen = 3.7 # center wavelength (Основная длина волны) 3.7-4.8 microns are goal
inc_fcen = 1 / inc_lcen  # center frequency (convert wavelength into frequency)
inc_df = 0.1 * inc_fcen  # frequency width
#depended
source_center = mp.Vector3(x=-0.5 * cell_X + in_pml,y=0,z=cell_Z/2)
flux_center = mp.Vector3(x=0.5 * cell_X - in_pml - 0.5 * in_gap,y=0, z=cell_Z/2)

# 3. General calculation settings
resolution = 30  # pixels/μm # At low resolution, less than 20 may not count
steps = 100 # depends on the size of the cell, with small cells you can start with 50, with large cells you need to increase