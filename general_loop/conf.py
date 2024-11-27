""""
Notice:
- all units in microns
- by default, any place where there are no objects there is air (ε=1)
 although this can be changed by setting the default_material variable.
-
-
"""

import meep as mp
import meep.materials as mat

# 1. Настройки для определения параметров мета-атома. Опиши форму своего мета-атома
# this is about PML
in_pml = 2.0  # Default = 1 micron. PML thickness （Опредяляет толщину абсолютно поглощающего слоя, который ограничивает зону вычисления)
in_sub = 1.0  # Depends on a wavelength = 2 microns. substrate thickness (Толщина подложки)
in_gap = 1.0  # Depends on idk = 1 microns. padding between grating and PML (Расстояние между мета-атомом и полностью поглощающим слоем)
in_material = mp.Medium(epsilon=2.0)

# this is about meta-atom
at_per = 5  # 5 microns. grating periodicity (Частота повторения ячейки). must be larger than meta-atom
at_material = mp.Medium(epsilon=2.0) # Для Si. https://refractiveindex.info/?shelf=main&book=Si&page=Li-293K. Потом заменю на библиотеку материалов.
at_hei = 2  # meta-atom height (equals Y axis in global axis)
at_W = 4 # long of meta-atom (equals X axis in global axis)
at_H = 3 # width of meta-atom (equals Z axis in global axis)
#depended
cell_X = in_pml + in_sub + at_hei + in_gap + in_pml # Size of whole simulation a cell.
cell_Y = at_per # Size of compute a cell. The Y size of cell simulation
cell_Z = at_H + in_gap # Size on compute a cell. The Z size of cell
sub_center = mp.Vector3(x=- 0.5 * at_hei, y=0, z=cell_Z/2)
at_center = mp.Vector3(0.5 * in_sub)

# 2. Настройки для определения источника излучения
inc_lcen = 3  # center wavelength (Основная длина волны)
inc_fcen = 1 / inc_lcen  # center frequency (MEEP работает только с частотами, поэтому перевод в частоту)
inc_df = 0.1 * inc_fcen  # frequency width (Ширина относительно пика)
#depended
source_center = mp.Vector3(x=-0.5 * cell_X + in_pml,y=0,z=cell_Z/2)
flux_center = mp.Vector3(x=0.5 * cell_X - in_pml - 0.5 * in_gap,y=0, z=cell_Z/2)

# 3. Общие настройки вычисления
resolution = 20  # pixels/μm
steps = 50