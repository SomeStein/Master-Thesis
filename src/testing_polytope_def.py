import numpy as np
import matplotlib.pyplot as plt

def generate_next_scalars(scalar_set, quantiles):
    new_scalar_sets = []
    for quant in quantiles:
        if quant + sum(scalar_set) <= 1:
            new_scalar_set = scalar_set.copy()
            new_scalar_set.append(quant)
            new_scalar_sets.append(new_scalar_set)
            
    return new_scalar_sets

def generate_scalars(n, quantiles):
    
    scalars = [[]]
    
    for _ in range(n-1):
        for i in range(len(scalars)-1,-1,-1):
            scalar_set = scalars.pop(i)
            new_scalar_sets = generate_next_scalars(scalar_set, quantiles)
            scalars.extend(new_scalar_sets)
            
    for i in range(len(scalars)-1,-1,-1):
        scalar_set = scalars.pop(i)
        new_scalar_set = scalar_set.copy()
        new_scalar_set.append(1 - sum(new_scalar_set))
        scalars.append(new_scalar_set)
        
    return scalars

def generate_convex_combinations(V, quantiles = [0,0.25,0.5,0.75,1]):
    
    vectors = []
    
    scalars = generate_scalars(len(V),quantiles)
    
    for scalar_set in scalars:
        scaled_vectors = [V[i]*scalar_set[i] for i in range(len(V))]
        summed_vectors = np.sum(scaled_vectors, axis=0)
        vectors.append(summed_vectors)
        
    return vectors
    
        
V = [np.array([0,0]),np.array([1,0]),np.array([0,1])]

V += [-v for v in V]

quantiles = [0.1*i for i in range(11)]


vectors = generate_convex_combinations(V, quantiles)

# Stack them into one array
vec_array = np.stack(vectors)

# Plot endpoints as scatter points
plt.scatter(vec_array[:, 0], vec_array[:, 1], color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Vector Endpoints")
plt.show()