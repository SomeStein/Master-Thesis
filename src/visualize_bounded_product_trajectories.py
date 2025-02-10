import numpy as np
import plotly.express as px

def generate_matrix(n, threshold = 1, exact = False):
    """Generate an n×n matrix with spectral radius < threshold."""
    A = np.random.randn(n, n)
    spectral_radius = max(abs(np.linalg.eigvals(A)))
    if spectral_radius >= threshold or exact:
        A /= spectral_radius / threshold  # Scale to ensure spectral radius < threshold
    return A

def generate_starting_vector(n):
    """Generate a random nonzero n-dimensional vector, set to [1, 0, ..., 0]."""
    v = np.zeros(n)
    v[0] = 1
    return v

def generate_matrix_products(matrices, v, k_max):
    """Compute all possible matrix products A_{i_k} * ... * A_{i_1} * v for k ≤ k_max."""

    results = []
    m = len(matrices)
    
    if k_max <= 0:
        return results
    
    # Dictionary to store results for each length (key: product indices, value: vector)
    prev_results = {tuple([i]): matrices[i] @ v for i in range(m)}  # k = 1
    results.extend([(product, indices) for indices, product in prev_results.items()])
    
    # Iterate over product lengths k = 2 ... k_max
    for k in range(2, k_max + 1):
        temp_results = {}  # Temporary dictionary for k-length products
        
        # Generate k-length products from (k-1)-length products
        for indices, product in prev_results.items():
            for i in range(m):  # Multiply with each matrix to extend the product
                new_indices = indices + (i,)  # Add the new index to the tuple
                temp_results[new_indices] = matrices[i] @ product  # Store the result
        
        prev_results = temp_results  # Store the k-length results
        results.extend([(product, indices) for indices, product in temp_results.items()])

    return results

# --- Parameters ---
n = 2        # Matrix size (nxn)
m = 2        # Number of different matrices
k_max = 10   # Maximum product length
threshold = 1  # Spectral radius upper bound

# --- Generate Data ---
matrices = [generate_matrix(n, threshold, True) for _ in range(m)]
v = generate_starting_vector(n)  # Starting vector

results = generate_matrix_products(matrices, v, k_max)

# --- Prepare for Plotting ---
x_values, y_values, transparency, hover_labels = [], [], [], []

for product, indices in results:
    x_values.append(product[0])  # First coordinate
    y_values.append(product[1])  # Second coordinate
    transparency.append(1 - (len(indices) / (k_max+2))) # More transparent for larger k
    hover_labels.append(f"Indices: {indices}")  # Tooltip text

# --- Interactive Scatter Plot (Plotly) ---
fig = px.scatter(
    x=x_values,
    y=y_values,
    opacity=transparency,  # Adjust transparency by k value (inverse of k)
    color_discrete_sequence=["blue"] * len(x_values),
    hover_name=hover_labels
)

# Mark starting vector in red (only once)
fig.add_scatter(x=[v[0]], y=[v[1]], mode="markers", marker=dict(color="red", size=10), name="Starting Vector")

fig.update_layout(
    title="Matrix Product Scatter Plot",
    xaxis_title="X",
    yaxis_title="Y",
    showlegend=True
)

fig.show()