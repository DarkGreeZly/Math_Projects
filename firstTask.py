import numpy as np


#first equation

# # Given matrix A
# A = np.array([[1, 0, 4],
#               [2, 3, 0],
#               [1, -1, 2]])
#
# # Define the polynomial coefficients
# coefficients = [2, 8, 8]
#
# # Define the function to evaluate the polynomial for a scalar or a matrix
# def evaluate_polynomial(matrix):
#     return np.polyval(coefficients, matrix)
#
# # Evaluate the polynomial for the matrix A
# result_matrix = evaluate_polynomial(A)
#
# print("Matrix A:")
# print(A)
# print("\nResult of f(A):")
# print(result_matrix)

#second equation

# # Given matrices A, B, and C
# A = np.array([[4, 5],
#               [-5, -6]])
#
# B = np.array([[5, 8],
#               [-2, -3]])
#
# C = np.array([[2, 4],
#               [-3, -4]])
#
# # Solve for X using the formula X = A_inv * C * B_inv
# A_inv = np.linalg.inv(A)
# B_inv = np.linalg.inv(B)
#
# X = np.dot(np.dot(A_inv, C), B_inv)
#
# print("Matrix X:")
# print(X)

#third equation

# Given matrix
# matrix = np.array([[-2, 3, 2, 1, 3],
#                    [1, -2, 1, -1, 0],
#                    [3, 1, -1, 0, -1],
#                    [2, 2, 2, 0, 2]])
#
# # Find the rank of the matrix
# rank = np.linalg.matrix_rank(matrix)
#
# print("Rank of the matrix:", rank)
#
# # If the rank is equal to the number of columns, the system is consistent
# if rank == matrix.shape[1]:
#     print("The system is consistent. It may have unique solutions or infinitely many solutions.")
#
#     # If the rank is equal to the number of columns, the system has a unique solution
#     if rank == matrix.shape[1]:
#         # Use the pseudo-inverse to solve the system
#         solution = np.linalg.pinv(matrix) @ np.zeros((matrix.shape[0], 1))  # Assuming the right-hand side is zero
#         print("Unique Solution:")
#         print(solution)
#
# # If the rank is less than the number of columns, the system is inconsistent and has no solution
# else:
#     print("The system is inconsistent and has no solution.")

#fourth equation using matrix method

# # Coefficient matrix A
# A = np.array([[3, 1, 1],
#               [1, 3, 1],
#               [1, 1, 3]])
#
# # Right-hand side column vector B
# B = np.array([4, 6, 0])
#
# # Solve the system of linear equations
# solution = np.linalg.solve(A, B)
#
# print("Solution:")
# print("x1 =", solution[0])
# print("x2 =", solution[1])
# print("x3 =", solution[2])

#fourth equation by Kramer's formulas
#
# # Coefficient matrix A
# A = np.array([[3, 1, 1],
#               [1, 3, 1],
#               [1, 1, 3]])
#
# # Right-hand side column vector B
# B = np.array([4, 6, 0])
#
# # Initialize the solution vector
# solution = np.zeros_like(B, dtype=float)
#
# # Calculate the determinant of the coefficient matrix A
# det_A = np.linalg.det(A)
#
# # Iterate over each variable and find its solution using Kramer's rule
# for i in range(len(B)):
#     A_i = A.copy()
#     A_i[:, i] = B
#
#     # Calculate the determinant of the modified matrix A_i
#     det_A_i = np.linalg.det(A_i)
#
#     # Calculate the solution for the i-th variable
#     solution[i] = det_A_i / det_A
#
# print("Solution:")
# for i, value in enumerate(solution):
#     print(f"x{i + 1} =", value)

#fifth equation

# import sympy as sp
# import math
#
# a=sp.Symbol('a')
# b=sp.Symbol('b')
#
# X = sp.expand((3*a+2*b)*(-a-3*b))
# X=X.evalf(subs={a:math.sqrt(3),b:1,a*b:math.cos(math.pi/6)})
# print("Result:")
# print(round(X,2))

# X1 = sp.expand(sp.sqrt((2*a-b)**2))
# X1 = X1.evalf(subs={a:math.sqrt(3),b:1,a*b:math.cos(math.pi/6)})
# print("Result: ")
# print(round(X1,2))
# # Define vectors a and b
# a = np.array([1, 2, 3])  # Replace a1, a2, a3 with actual components of vector a
# b = np.array([4, 5, 6])  # Replace b1, b2, b3 with actual components of vector b
#
# # Given expression
# expression = np.dot(3 * a + 2 * b, -a + 3 * b)
#
# # Expand and simplify the expression
# result = np.expand_dims(expression, axis=0)
#
# # Display the result
# print("Result:")
# print(result)

#sixth equation

import numpy as np

# Define the coordinates of points A, B, C, and D
A = np.array([1, 0, 4])
B = np.array([2, -2, -1])
C = np.array([3, -1, 0])
D = np.array([3, 2, 5])

# Vectors AB, AC, and AD
AB = B - A
AC = C - A
AD = D - A

# Area of the triangle ABC (magnitude of the cross product of AB and AC)
area_ABC = 0.5 * np.linalg.norm(np.cross(AB, AC))

# Volume of the pyramid ABCD (scalar triple product of AB, AC, and AD)
volume_ABCD = abs(np.dot(AB, np.cross(AC, AD))) / 6

print("Area of the face ABC:", area_ABC)
print("Volume of the pyramid ABCD:", volume_ABCD)


