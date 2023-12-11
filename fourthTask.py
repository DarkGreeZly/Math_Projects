import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Створення функції, що малює криву, яку ви хочете анімувати
# def draw_curve():
# # Отримання точок для кривої (можна замінити цей рядок на вашу криву)
#     x = np.linspace(0, 10, 100)
#     y = np.sin(x)
#
# # Малюємо криву
#     plt.plot(x, y, color='blue')  # Встановлення початкового кольору
#     plt.show()
#
# # Виклик функції, щоб перевірити, чи малюється крива правильно
# draw_curve()
#
# # Ініціалізація графіку
# fig, ax = plt.subplots()
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
# line, = ax.plot(x, y, color='blue')
#
# # Функція оновлення кадрів для анімації
# def update(frame):
#     line.set_ydata(np.sin(x + frame * 0.1))  # Рух кривої
#     if frame % 10 == 0:  # Зміна кольору кожні 10 кадрів
#         line.set_color(np.random.rand(3,))
#         return line,
#
# # Створення анімації
# ani = animation.FuncAnimation(fig, update, frames=100, interval=50)
#
# # Відображення анімації
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# def generate_beautiful_rose(theta, n):
#     r = 3 * np.cos(n * theta) * np.exp(-0.1 * theta)
#     return r
#
# def update(frame):
#     ax.clear()
#     theta = np.linspace(0, 2 * np.pi, 1000)
#     r = generate_beautiful_rose(theta + np.radians(frame), n_petal)
#     ax.plot(theta, r, color='salmon', linewidth=2, linestyle='-', alpha=0.7)
#     ax.set_title(f'Beautiful n-petal Rose Animation - Frame {frame}')
#
# n_petal = 6  # Adjust the number of petals for a more beautiful rose
#
# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
# ax.set_facecolor('#F0F0F0')  # Set a light gray background
#
# frames = 200
# animation = FuncAnimation(fig, update, frames=frames, interval=50, repeat=True)
#
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


# Function to generate the n-petal rose in polar coordinates
def generate_rose(theta, n):
    r = 3 * np.cos(n * theta)
    return r


# Function to generate a surface based on time
def generate_surface(theta, time):
    x = theta
    y = theta
    z = np.sin(np.sqrt(x ** 2 + y ** 2) + time)
    return x, y, z


# Function to update the plot for each frame of the animation
def update(frame):
    ax.cla()

    # Plotting wireframe surface
    theta1 = np.linspace(0, 2 * np.pi, 100)
    radius1 = np.linspace(0, 5, 100)
    Theta1, Radius1 = np.meshgrid(theta1, radius1)
    X1, Y1, Z1 = polar_surface(Radius1, Theta1)
    ax.plot_wireframe(X1, Y1, Z1, color='blue', label='Wireframe Surface')

    # Plotting animated surface
    theta = np.linspace(-5, 5, 100)
    x, y, z = generate_surface(theta, frame / 10)
    surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='black', alpha=0.7)
    ax.set_title(f'Surface Evolution - Frame {frame}')

    # Adding labels and legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)


# Create a figure and axis
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(131, projection='3d')

# Set animation parameters
frames = 100
animation = FuncAnimation(fig, update, frames=frames, interval=50, repeat=True)

# Show the animation
plt.show()
