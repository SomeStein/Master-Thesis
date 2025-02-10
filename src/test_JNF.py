import numpy as np
import sympy as sp

# Matrix size
n = 30

# Define Jordan blocks
block_sizes = [10, 8, 6, 4, 2]  # Sum must be 30
lambda_val = 2

# Create Jordan matrix
J = np.zeros((n, n))
start_idx = 0

for size in block_sizes:
    J[start_idx:start_idx + size, start_idx:start_idx + size] = np.eye(size) * lambda_val
    for i in range(size - 1):
        J[start_idx + i, start_idx + i + 1] = 1  # Superdiagonal ones
    start_idx += size

# Generate a random invertible matrix P
while True:
    P = np.random.rand(n, n)
    if np.linalg.det(P) != 0:  # Ensure it's invertible
        break

# Convert to symbolic matrix
J_matrix = sp.Matrix(J)
P_matrix = sp.Matrix(P)

# Define symbolic variables
t, k = sp.symbols('t k')
power_terms = [k * t**(k-1) * sp.exp(lambda_val * t) for k in range(1, 11)]
power_terms_matrix = sp.Matrix(power_terms)

# Display results
sp.pprint(J_matrix)
sp.pprint(P_matrix)
sp.pprint(power_terms_matrix)