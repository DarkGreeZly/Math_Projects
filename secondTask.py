import numpy as np
import matplotlib.pyplot as plt

#first graphic

# # Define the parametric functions
# def x_parametric(t):
#     return np.tan(2 * np.exp(-t))
#
# def y_parametric(t):
#     return np.log(np.cos(1 / np.tan(np.exp(t))))
#
# # Generate values for the parameter t
# t_values = np.linspace(-2, 2, 1000)
#
# # Calculate corresponding x and y values
# x_values = x_parametric(t_values)
# y_values = y_parametric(t_values)
#
# # Plot the parametric curve
# plt.plot(x_values, y_values, label='Curve', color='blue')
#
# # Plot the graph (y = ln(cot(e^t)), x = tan(2e^-t))
# graph_x = np.linspace(-2, 2, 1000)
# graph_y = np.log(np.cos(1 / np.tan(np.exp(graph_x))))
# plt.plot(graph_x, graph_y, label='Graph', linestyle='--', color='red')
#
# # Add labels and legend
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Parametrically Defined Curve')
# plt.legend()
#
# # Sign the graph and the curve
# plt.text(0.5, 2, 'Curve', color='blue')
# plt.text(0.5, -2, 'Graph', color='red')
#
# # Show the plot
# plt.grid(True)
# plt.show()

# second graphic
#
#
# # Define the polar equation
# def polar_equation(gamma):
#     return 5 / (1 - 0.5 * np.sin(gamma))
#
# # Generate values for gamma
# gamma_values = np.linspace(0, 2 * np.pi, 1000)
#
# # Calculate corresponding radial values using the polar equation
# p_values = polar_equation(gamma_values)
#
# # Plot the polar curve
# plt.polar(gamma_values, p_values, label=r'$p = \frac{5}{1 - \frac{1}{2} \sin(\gamma)}$')
#
# # Add labels and legend
# plt.title('Polar Curve')
# plt.legend()
#
# # Show the plot
# plt.show()

#third graphic
#
# # Given curve equation in Cartesian coordinates
# def cartesian_equation(x, y):
#     return 9 * x**2 - 24 * x * y + 16 * y**2 - 20 * x + 110 * y - 50
#
# # Rotation matrix for counterclockwise rotation by angle alpha
# def rotation_matrix(alpha):
#     alpha_rad = np.radians(alpha)
#     return np.array([[np.cos(alpha_rad), -np.sin(alpha_rad)],
#                      [np.sin(alpha_rad), np.cos(alpha_rad)]])
#
# # Generate points in Cartesian coordinates
# x_values = np.linspace(-10, 10, 400)
# y_values = np.linspace(-10, 10, 400)
# x_grid, y_grid = np.meshgrid(x_values, y_values)
#
# # Transform points using the rotation matrix
# alpha = 63  # Angle in degrees
# rotation_matrix_alpha = rotation_matrix(alpha)
# x_rotated, y_rotated = np.einsum('ij,jkl->ikl', rotation_matrix_alpha, np.stack([x_grid, y_grid]))
#
# # Evaluate the equation in rotated coordinates
# curve_values_rotated = cartesian_equation(x_rotated, y_rotated)
#
# # Plot the curve at angle alpha
# plt.contour(x_grid, y_grid, curve_values_rotated, levels=[0], colors='blue', label=f'α = {alpha}°')
#
# # Add labels and legend
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Curve at Angle α')
# plt.legend()
#
# # Show the plot
# plt.grid(True)
# plt.show()

#fourth graphic first part

# # Define the functions
# def f(x):
#     return 2 / (x**2 - 1)
#
# def g(x):
#     return np.sin(x**2 + 1)
#
# # Generate x values
# x_values = np.linspace(-3, 3, 400)
#
# # Generate y values for each function
# y_f = f(x_values)
# y_g = g(x_values)
#
# # Plot the graphs
# plt.figure(figsize=(10, 5))
#
# # Plot the graph of f(x)
# plt.subplot(1, 2, 1)
# plt.plot(x_values, y_f, color='blue', label='f(x) = 2/(x^2 - 1)')
# plt.title('Graph of f(x)')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.legend()
#
# # Plot the graph of g(x)
# plt.subplot(1, 2, 2)
# plt.plot(x_values, y_g, color='brown', label='g(x) = sin(x^2 + 1)')
# plt.title('Graph of g(x)')
# plt.xlabel('x')
# plt.ylabel('g(x)')
# plt.legend()
#
# # Adjust layout for better visualization
# plt.tight_layout()
#
# # Show the plots
# plt.show()

#fourth graphic second part


# # Define the functions
# def f(x):
#     return 2 / (x**2 - 1)
#
# def g(x):
#     return np.sin(x**2 + 1)
#
# # Generate x values
# x_values = np.linspace(-3, 3, 400)
#
# # Generate y values for each function
# y_f = f(x_values)
# y_g = g(x_values)
#
# # Plot the graphs
# plt.figure(figsize=(8, 6))
#
# # Plot the graph of f(x) in blue with a thicker line
# plt.plot(x_values, y_f, color='blue', label=r'$f(x) = \frac{2}{x^2 - 1}$', linewidth=2)
#
# # Plot the graph of g(x) in brown with a thinner line
# plt.plot(x_values, y_g, color='brown', label=r'$g(x) = \sin(x^2 + 1)$', linewidth=1)
#
# # Draw the x and y axes through the origin
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
#
# # Add labels to the x and y axes
# plt.xlabel('x')
# plt.ylabel('y')
#
# # Add a legend
# plt.legend()
#
# # Set plot limits for better visualization
# plt.ylim([-2, 2])
#
# # Show the plot
# plt.grid(True)
# plt.title('Graphs of f(x) and g(x)')
# plt.show()

#fifth graphic

# # Ellipse equation: (x^2/18) + (y^2/8) = 1
# def ellipse(x):
#     return np.sqrt(8 - 8 * (x**2 / 18))
#
# # Straight line equation: 5x + y + 1 = 0
# def line(x):
#     return -5 * x - 1
#
# # Generate x values for the ellipse
# x_ellipse = np.linspace(-np.sqrt(18), np.sqrt(18), 400)
# y_ellipse_positive = ellipse(x_ellipse)
# y_ellipse_negative = -y_ellipse_positive
#
# # Generate x values for the line
# x_line = np.linspace(-4, 4, 400)
# y_line = line(x_line)
#
# # Plot the graphs
# plt.figure(figsize=(8, 6))
#
# # Plot the ellipse in blue with a solid line
# plt.plot(x_ellipse, y_ellipse_positive, color='blue', label='Ellipse (Upper)', linestyle='-', linewidth=2)
# plt.plot(x_ellipse, y_ellipse_negative, color='blue', label='Ellipse (Lower)', linestyle='-', linewidth=2)
#
# # Plot the line in red with a dashed line
# plt.plot(x_line, y_line, color='red', label='Line', linestyle='--', linewidth=2)
#
# # Draw the x and y axes
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
#
# # Add labels to the x and y axes
# plt.xlabel('x')
# plt.ylabel('y')
#
# # Add a legend
# plt.legend()
#
# # Set equal aspect ratio for better visualization
# plt.gca().set_aspect('equal', adjustable='box')
#
# # Show the plot
# plt.grid(True)
# plt.title('Ellipse and Line in the Same Plot')
# plt.show()


#sixth graphic


# # Set a random seed for reproducibility
# np.random.seed(42)
#
# # 1. Bar Graph
# categories = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
# colors = ['red', 'green', 'blue', 'purple']
#
# bar_values = np.random.randint(1, 10, len(categories))
# bar_width = 0.7
#
# plt.bar(categories, bar_values, color=colors)
# plt.title('Bar Graph')
# plt.xlabel('Categories')
# plt.ylabel('Values')
# plt.show()
#
# # 2. Histogram with Groupings
# data = np.random.randn(1000) * 3 + 10
#
# plt.hist(data, bins=20, color='orange', edgecolor='black')
# plt.title('Histogram with Groupings')
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.show()
#
# # 3. Pie Chart with Sectors
# labels = ['Sector 1', 'Sector 2', 'Sector 3', 'Sector 4', 'Sector 5', 'Sector 6']
# sizes = [15, 10, 25, 20, 5, 10]
#
# explode = (0, 0, 0.1, 0, 0, 0)  # Explode the third sector
#
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['cyan', 'magenta', 'yellow', 'lime', 'blue', 'purple'], explode=explode)
# plt.title('Pie Chart with Sectors')
# plt.show()
#
# # 4. Scatter Plot with Various Symbols
# num_points = 50
# x = np.random.rand(num_points)
# y = np.random.rand(num_points)
# symbols = ['o', 's', 'D', '^', 'v']
#
# plt.scatter(x, y, c='red', marker=np.random.choice(symbols, num_points))
# plt.title('Scatter Plot with Various Symbols')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Generate random data for the scatter plot
np.random.seed(42)  # Setting seed for reproducibility
x1 = np.random.rand(50)  # Random x values between 0 and 1
y1 = np.random.rand(50)  # Random y values between 0 and
x2 = np.random.rand(50)
y2 = np.random.rand(50)

# Create a scatter plot
plt.scatter(x1, y1, marker="^", color='blue', label='Triangles')
plt.scatter(x2, y2, marker="*", color='red', label='Stars')


# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot Example')

# Add a legend
plt.legend()

# Show the plot
plt.show()

