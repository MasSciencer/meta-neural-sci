import numpy as np
import matplotlib.pyplot as plt

# Параметры метаатомов
angles = np.linspace(0, 180, 10)  # Углы ориентации (в градусах)
wavelength = 1550e-9  # Длина волны (в метрах)
phase_shifts = 2 * np.radians(angles)  # Геометрическая фаза, φ = 2θ

# Визуализация структуры метаатомов
def plot_meta_atoms(angles, phase_shifts):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # График ориентации метаатомов
    ax[0].scatter(angles, phase_shifts, color='blue', label='Phase shift')
    ax[0].plot(angles, phase_shifts, linestyle='--', color='red')
    ax[0].set_title("Фазовый сдвиг метаатомов")
    ax[0].set_xlabel("Угол ориентации (градусы)")
    ax[0].set_ylabel("Фазовый сдвиг (радианы)")
    ax[0].legend()
    ax[0].grid()

    # Визуализация метаатомов
    for angle in angles:
        x = [0, np.cos(np.radians(angle))]
        y = [0, np.sin(np.radians(angle))]
        ax[1].plot(x, y, label=f"{angle:.1f}°")
    ax[1].set_title("Ориентация метаатомов")
    ax[1].set_aspect('equal')
    ax[1].set_xlim(-1.5, 1.5)
    ax[1].set_ylim(-1.5, 1.5)
    ax[1].grid()

    plt.show()

# Вызов функции визуализации
plot_meta_atoms(angles, phase_shifts)

# Пример применения фазового сдвига
def calculate_wavefront_modulation(angles, phase_shifts):
    """
    Функция для демонстрации, как фазовые сдвиги используются для управления волновым фронтом.
    """
    incident_wave = np.exp(1j * 2 * np.pi / wavelength)  # Падающая волна
    modulated_wave = incident_wave * np.exp(1j * phase_shifts)  # С учётом фазового сдвига
    return modulated_wave

# Расчет и визуализация результирующей волны
modulated_wave = calculate_wavefront_modulation(angles, phase_shifts)
plt.figure(figsize=(8, 5))
plt.plot(angles, np.angle(modulated_wave), label="Модулированный фазовый сдвиг")
plt.title("Модуляция волнового фронта")
plt.xlabel("Угол ориентации метаатомов (градусы)")
plt.ylabel("Фазовый сдвиг (радианы)")
plt.grid()
plt.legend()
plt.show()
