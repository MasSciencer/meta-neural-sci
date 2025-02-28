import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметры задачи
wavelength = 3.7e-6  # длина волны, м
focal_length = 50e-3  # фокусное расстояние, м (50 mm)
num_points = 1920  # количество точек для расчётов
aperture_diameter_initial = 5e-3 # 5 mm
aperture_diameter_final = 50e-3 # 50 mm
nunber_frames = 600

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
colorbar = plt.colorbar(cax, label='Фазовый сдвиг (0 - 2π)')

plt.title('Фазовый сдвиг для фокусировки света')
plt.xlabel('x (мкм)')
plt.ylabel('y (мкм)')
plt.axis('equal')

def update(frame):
    global cax  # Доступ к глобальному объекту cax
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
    ax.set_title(f'Фазовый сдвиг для фокусировки света (диаметр апертуры = {frame * 1e6:.1f} мкм)')

# Анимация
ani = FuncAnimation(fig, update, frames=np.linspace(aperture_diameter_initial, aperture_diameter_final, nunber_frames), repeat=True, interval = 40)
ani.save('animation.mp4')

plt.show()
