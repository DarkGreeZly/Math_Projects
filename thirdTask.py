import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#first 3-d figures

# # Create a 3D figure
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot a semicircle
# theta = np.linspace(0, np.pi, 100)
# x_s = np.cos(theta)
# y_s = np.sin(theta)
# z_s = np.zeros_like(theta)
# ax.plot(x_s, y_s, z_s, color='red', label='Semicircle')
#
# # Plot an arrow
# ax.quiver(0, 0, 0, 1, 1, 1, color='green', label='Arrow')
#
# # Plot a triangle
# triangle_vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3)/2, 0]])
# triangle_vertices = np.vstack([triangle_vertices, triangle_vertices[0]])  # Close the triangle
# ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], triangle_vertices[:, 2], color='blue', label='Triangle')
#
# # Plot a sector
# theta_sector = np.linspace(0, np.pi/2, 100)
# x_sec = np.cos(theta_sector)
# y_sec = np.sin(theta_sector)
# z_sec = np.zeros_like(theta_sector)
# ax.plot(x_sec, y_sec, z_sec, color='purple', label='Sector')
#
# # Plot a rhombus
# rhombus_vertices = np.array([[0, 0, 0], [1, 0, 0], [1.5, 1, 0], [0.5, 1, 0]])
# rhombus_vertices = np.vstack([rhombus_vertices, rhombus_vertices[0]])  # Close the rhombus
# ax.plot(rhombus_vertices[:, 0], rhombus_vertices[:, 1], rhombus_vertices[:, 2], color='orange', label='Rhombus')
#
# # Plot a rectangle
# rectangle_vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
# ax.plot(rectangle_vertices[:, 0], rectangle_vertices[:, 1], rectangle_vertices[:, 2], color='cyan', label='Rectangle')
#
# # Add labels and legend
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# ax.legend()
#
# # Show the 3D plot
# plt.show()

# second 3-d figure

# # Function to generate N-polygon vertices
# def generate_polygon(N, radius=1):
#     theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
#     x = radius * np.cos(theta)
#     y = radius * np.sin(theta)
#     return x, y
#
# # Create a 3D figure
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Parameters for the N-polygon
# N = 18
# radius = 1
#
# # Generate N-polygon vertices
# x, y = generate_polygon(N, radius)
# z = np.zeros_like(x)
#
# # Plot the N-polygon
# polygon = Poly3DCollection([list(zip(x, y, z))], color='cyan', alpha=0.5, edgecolors='black', linewidths=1.5)
# ax.add_collection3d(polygon)
#
# # Add arrows to outline the N-polygon
# for i in range(N):
#     ax.quiver(x[i], y[i], z[i], x[(i + 1) % N] - x[i], y[(i + 1) % N] - y[i], z[(i + 1) % N] - z[i],
#               color='black', arrow_length_ratio=0.1, linewidth=1)
#
# # Add text "Hello world!" in the middle
# ax.text(0, 0, 0, 'Hello world!', color='red', fontsize=12, ha='center', va='center')
#
# # Set equal aspect ratio for better visualization
# ax.set_box_aspect([1, 1, 1])
#
# # Add labels and title
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# ax.set_title('Colored N-polygon with Arrows and Text')
#
# # Show the 3D plot
# plt.show()

#third 3-d figure

# # Create a 3D figure
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Generate data for the surface
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# x, y = np.meshgrid(x, y)
# z = np.sin(np.sqrt(x**2 + y**2))
#
# # Plot the surface
# surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='black', linewidth=0.5)
#
# # Add a label to the surface plot
# ax.text2D(0.05, 0.95, 'Surface Plot with Label', transform=ax.transAxes, fontsize=12, color='red')
#
# # Add labels and title
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# ax.set_title('3D Surface Plot')
#
# # Show the 3D plot
# plt.show()

#fourth 3-d figure


# # Create a 3D figure
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Generate data for the surface
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# x, y = np.meshgrid(x, y)
# z = np.sqrt(x**2 + 3*y**2 + 4*x - 1) + 2
#
# # Plot the surface
# surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='black', linewidth=0.5, alpha=0.7)
#
# # Add a label to the surface plot
# ax.text2D(0.05, 0.95, 'Colored Surface Plot', transform=ax.transAxes, fontsize=12, color='red')
#
# # Add labels and title
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# ax.set_title('3D Surface Plot')
#
# # Show the 3D plot
# plt.show()

#fifth 3-d figure


# # Create a 3D figure
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Number of columns
# num_columns = 18 + 4
#
# # Heights for each column (random heights using np.random)
# heights = np.random.rand(num_columns)
#
# # Plot the first set of columns at x=1
# ax.bar(np.ones(num_columns), heights, zs=0, zdir='y', color='blue', alpha=0.7, width=0.5, label='Columns at x=1')
#
# # Plot the second set of columns at x=3
# ax.bar(3 * np.ones(num_columns), heights, zs=0, zdir='y', color='green', alpha=0.7, width=0.5, label='Columns at x=3')
#
# # Plot the third set of columns varying from y=0.05 to 1
# y_values = np.linspace(0.05, 1, num_columns)
# ax.bar(np.random.uniform(2, 2.5, num_columns), heights, zs=y_values, zdir='y', color='orange', alpha=0.7, width=0.5, label='Columns in y-plane')
#
# # Set labels and title
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# ax.set_title('Spatial Diagrams with Columns')
#
# # Add a legend
# ax.legend()
#
# # Show the 3D plot
# plt.show()


#sixth 3-d figure


# # Function to generate the surface in polar coordinates
# def polar_surface(radius, theta):
#     x = radius * np.cos(theta)
#     y = radius * np.sin(theta)
#     z = 3 * np.cos(3 * theta)
#     return x, y, z
#
# # Function to generate the filled contour surface
# def filled_contour_surface(x, y, z):
#     return (x**2 / 9) - (y**2 / 4) + (z**2 / 16)
#
# # Create a 3D figure with three subplots
# fig = plt.figure(figsize=(15, 5))
#
# # First subplot: Wireframe surface plot
# ax1 = fig.add_subplot(131, projection='3d')
# theta1 = np.linspace(0, 2 * np.pi, 100)
# radius1 = np.linspace(0, 5, 100)
# Theta1, Radius1 = np.meshgrid(theta1, radius1)
# X1, Y1, Z1 = polar_surface(Radius1, Theta1)
# ax1.plot_wireframe(X1, Y1, Z1, color='blue', label='Wireframe Surface')
# ax1.set_xlabel('X-axis')
# ax1.set_ylabel('Y-axis')
# ax1.set_zlabel('Z-axis')
# ax1.set_title('Wireframe Surface Plot')
# ax1.legend()
#
# # Second subplot: Scattered surface plot (modify based on your data)
# ax2 = fig.add_subplot(132, projection='3d')
# x2 = np.random.rand(50)
# y2 = np.random.rand(50)
# z2 = np.random.rand(50)
# ax2.scatter(x2, y2, z2, color='green', label='Scattered Surface')
# ax2.set_xlabel('X-axis')
# ax2.set_ylabel('Y-axis')
# ax2.set_zlabel('Z-axis')
# ax2.set_title('Scattered Surface Plot')
# ax2.legend()
#
# # Third subplot: Filled contour plot in polar coordinates
# ax3 = fig.add_subplot(133, projection='3d')
# theta3 = np.linspace(0, 2 * np.pi, 100)
# radius3 = np.linspace(0, 5, 100)
# Theta3, Radius3 = np.meshgrid(theta3, radius3)
# X3, Y3, Z3 = polar_surface(Radius3, Theta3)
# contour_data = filled_contour_surface(X3, Y3, Z3)
# contour = ax3.contourf(X3, Y3, Z3, contour_data, cmap='viridis', levels=20, alpha=0.7)
# ax3.set_xlabel('X-axis')
# ax3.set_ylabel('Y-axis')
# ax3.set_zlabel('Z-axis')
# ax3.set_title('Filled Contour Plot in Polar Coordinates')
# fig.colorbar(contour, ax=ax3, label='Contour Levels')
#
# # Adjust layout for better visualization
# plt.tight_layout()
#
# # Show the plot
# plt.show()
