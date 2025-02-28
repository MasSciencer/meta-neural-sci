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
in_sub = 1.0  # Depends on a wavelength = 1 microns. substrate thickness (Толщина подложки)
in_gap1 = 0.5  # Depends on idk = 1 microns. padding between grating and PML (Расстояние между мета-атомом и полностью поглощающим слоем)
in_gap2 = 2.0
in_material = mp.Medium(epsilon=6.5) # n≈sqrl(epsilon) for a wavelength of 3.7 microns, the refractive index is 3.4 for amorphous silicon
"""
На длине волны 3.7 мкм значения диэлектрической проницаемости (эпсилон) для сапфира (Al₂O₃) и селенид цинка (ZnSe) следующие:

Сапфир (Al₂O₃):

Реальная часть эпсилона: около 3.1-3.2.
Мнимая часть эпсилона: очень малая, что делает сапфир почти без потерь в ИК-диапазоне. Это хорошее свойство для использования в оптических компонентах.
Селенид цинка (ZnSe):

Реальная часть эпсилона: примерно 6.5-7.
Мнимая часть эпсилона также мала, что указывает на низкие оптические потери и хорошую прозрачность в ИК-диапазоне, что делает ZnSe идеальным для использования в качестве активного материала.
Таким образом, оба материала имеют хорошие оптические характеристики для работы в диапазоне 3.7 мкм, при этом ZnSe имеет несколько более высокую диэлектрическую проницаемость, что может быть полезно для фокусировки и манипуляции фазами в метаматериале.
"""

# this is about meta-atom
at_per = 5  # 10 microns. grating periodicity (Частота повторения ячейки). must be larger than meta-atom
at_material = mp.Medium(epsilon=12) # Для Si. https://refractiveindex.info/?shelf=main&book=Si&page=Li-293K. Потом заменю на библиотеку материалов.
at_hei = 3  # meta-atom height (equals Y axis in global axis)
#depended
cell_X = in_pml + in_gap1 + in_sub + at_hei + in_gap2 + in_pml # Size of whole simulation a cell.
cell_Y = at_per # Size of compute a cell. The Y size of cell simulation
sub_center = mp.Vector3(x=-0.5*cell_X+in_pml+in_gap1+0.5*in_sub)
at_center = mp.Vector3(x=-0.5*cell_X+in_pml+in_gap1+in_sub+at_hei*0.5)
k_point = mp.Vector3(0, 0, 0)
# 2. Settings for determining the radiation source
inc_lcen = 3.7 # center wavelength (Основная длина волны) 3.7-4.8 microns are goal
inc_fcen = 1 / inc_lcen  # center frequency (convert wavelength into frequency)
inc_df = 0.2 * inc_fcen  # frequency width
#depended
source_center = mp.Vector3(x=-0.5*cell_X+in_pml)
flux_center = mp.Vector3(x=-0.5*cell_X+in_pml+in_gap1+in_sub+at_hei+0.5*in_gap2)

print(f"substrait begin in {sub_center.x-0.5*in_sub} and end {sub_center.x + 0.5*in_sub}")
print(f"atom begin in {at_center.x-0.5*at_hei} and end {at_center.x + 0.5*at_hei}")
print(f"flux_center begin in {flux_center.x}")

# 3. General calculation settings
resolution = 40  # pixels/μm # At low resolution, less than 20 may not count
steps = 100 # depends on the size of the cell, with small cells you can start with 50, with large cells you need to increase