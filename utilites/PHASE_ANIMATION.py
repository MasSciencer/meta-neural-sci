import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметры задачи
wavelength = 4.25e-6  # длина волны, м
focal_length = 5e-3  # фокусное расстояние, м (5 mm)
num_points = 1920  # количество точек для расчётов
aperture_diameter_initial = 0.2e-3 # 0.2 mm
aperture_diameter_final = 5e-3 # 50 mm
number_frames = 80
global_state = 0

fig, ax = plt.subplots(figsize=(12.5, 10))

# Начальные значения для координат и фазового сдвига
x = np.linspace(-aperture_diameter_initial / 2, aperture_diameter_initial / 2, num_points)
y = np.linspace(-aperture_diameter_initial / 2, aperture_diameter_initial / 2, num_points)
x, y = np.meshgrid(x, y)
r = np.sqrt(x**2 + y**2)

# Инициализация фазового сдвига
initial_phase_shift = (2 * np.pi / wavelength) * (np.sqrt(r**2 + focal_length**2) - focal_length)
initial_phase_shift = np.mod(initial_phase_shift, 2 * np.pi)

# Отображение начального состояния
cax = ax.pcolormesh(x * 1e6, y * 1e6, initial_phase_shift, shading='auto', cmap='jet')
colorbar = plt.colorbar(cax, label='Phase shift (0 - π)')

plt.title('The distribution of phase shifts for different lens diameters at a fixed focal length')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.axis('equal')

def update(frame):
    global cax  # Доступ к глобальному объекту cax
    global global_state
    aperture_diameter = frame  # изменяем диаметр апертуры с каждым кадром
    x = np.linspace(-aperture_diameter / 2, aperture_diameter / 2, num_points)
    y = np.linspace(-aperture_diameter / 2, aperture_diameter / 2, num_points)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)

    aperture_mask = r <= (aperture_diameter / 2)
    phase_shift = (2 * np.pi / wavelength) * (np.sqrt(r**2 + focal_length**2) - focal_length)
    phase_shift = np.mod(phase_shift, 2 * np.pi)
    phase_shift *= aperture_mask

    # Удаляем старый объект и создаём новый
    cax.remove()
    cax = ax.pcolormesh(x * 1e6, y * 1e6, phase_shift, shading='auto', cmap='jet')

    # Обновляем границы осей
    ax.set_xlim([x.min() * 1e6, x.max() * 1e6])
    ax.set_ylim([y.min() * 1e6, y.max() * 1e6])
    ax.set_title(f'Phase distribution for an aperture  = {frame * 1e6:.1f} μm)')
    global_state += 1
    print(f'global state = {global_state}')

# Анимация
ani = FuncAnimation(fig, update, frames=np.linspace(aperture_diameter_initial, aperture_diameter_final, number_frames), repeat=True, interval = 120)
ani.save('animation.mp4')

plt.show()
