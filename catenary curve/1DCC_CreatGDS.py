import numpy as np
from scipy.integrate import quad
import gdspy
import matplotlib.pyplot as plt

M = 40 # число радиальных слоёв.
N = 6 # число вертикальных повторений для каждого слоя.
w = 0.5 # 1000 nm (расстояние промежутка), оно должно быть минимальным и определяется точностью изготовления...или нет. Нужен рисерч
wavelength = 3.7 # micrions
k = 2 * np.pi / wavelength #волновое число
f = 200 # focal length 10mm
Q = 100 #детализация
ymax_limit = 500 # Ограничение по вертикали для одной кривой, это нужно, чтобы кривые первых порядков не были монструозно большими

# Prepare for visualization
fig, ax = plt.subplots(figsize=(10, 2))

# Initialize GDSII library and structure
lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
structure = gdspy.Cell('CATENARY')

# Precompute radii
rt = np.sqrt(4 * f * (-np.pi / 2 + np.arange(1, M+1) * np.pi) / k)


# Define the function to integrate
def fun(x):
    return np.tan((k * x ** 2) / (4 * f))


# Main loop for layers
for m in range(M):
    rt0 = 0 if m == 0 else rt[m - 1] + 0.1
    x = np.linspace(rt0, rt[m] - 0.01, Q)

    # Compute y values by integration
    y = np.array([quad(fun, rt0, xi)[0] for xi in x])

    for n in range(N+1):
        xt = x.copy()
        ip = np.argmin(y)
        y1 = y - y[ip]
        y11 = y1.copy()
        ymax = 0.5 * (np.max(x) - np.min(x))
        if (ymax) >= ymax_limit: ymax = ymax_limit

        # Apply constraints to y1
        y1[y1 > ymax] = ymax
        y2 = y11 - w
        y2[y2 > ymax] = ymax

        # Filter out points exceeding ymax
        mask = y2 <= ymax
        xt = xt[mask]
        y1 = y1[mask]
        yy2 = y2[mask]

        # Generate the boundary coordinates
        xx = np.concatenate([xt, xt[::-1]])
        yy = np.concatenate([y1, y2[::-1]]) + 2 * n * w
        xy = np.vstack([xx, yy]).T

        # Add the boundary to the GDSII structure
        structure.add(gdspy.Polygon(xy, layer=1))

        # Plot for visualization
        ax.plot(xx, yy, color='green', linewidth=0.5)

        # Mirror the layers for negative y-values
        xx = -np.concatenate([xt, xt[::-1]])
        yy = -np.concatenate([y1, y2[::-1]]) + 2 * n * w + w
        xy = np.vstack([xx, yy]).T

        structure.add(gdspy.Polygon(xy, layer=1))

        # Plot for visualization
        ax.plot(xx, yy, color='blue', linewidth=0.5)


# Write the GDSII library to file
lib.add(structure)
lib.write_gds('catenary1dfocus.gds')

# Finalize the plot
ax.set_aspect('equal', adjustable='datalim')
ax.set_title('Catenary Lens Structure')
ax.set_xlabel('x (microns)')
ax.set_ylabel('y (microns)')
plt.grid(True)
plt.show()
