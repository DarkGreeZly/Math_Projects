import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Створення функції, що малює криву, яку ви хочете анімувати
def draw_curve():
# Отримання точок для кривої (можна замінити цей рядок на вашу криву)
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

# Малюємо криву
    plt.plot(x, y, color='blue')  # Встановлення початкового кольору
    plt.show()

# Виклик функції, щоб перевірити, чи малюється крива правильно
draw_curve()

# Ініціалізація графіку
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y = np.sin(x)
line, = ax.plot(x, y, color='blue')

# Функція оновлення кадрів для анімації
def update(frame):
    line.set_ydata(np.sin(x + frame * 0.1))  # Рух кривої
    if frame % 10 == 0:  # Зміна кольору кожні 10 кадрів
        line.set_color(np.random.rand(3,))
        return line,

# Створення анімації
ani = animation.FuncAnimation(fig, update, frames=100, interval=50)

# Відображення анімації
plt.show()