import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def legendre_polynomial(n):
   
    if n == 0:
        return np.poly1d([1])  # P_0(x) = 1
    elif n == 1:
        return np.poly1d([1, 0])  # P_1(x) = x

    P_0 = np.poly1d([1])
    P_1 = np.poly1d([1, 0])

    for i in range(1, n):
        P_next = ((2 * i + 1) * np.poly1d([1, 0]) * P_1 - i * P_0) / (i + 1)
        P_0, P_1 = P_1, P_next

    return P_1


def legendre_companion_matrix(n):

    poly = legendre_polynomial(n)
    coeffs = poly.coeffs  
    
    coeffs = coeffs / coeffs[0]

    companion_matrix = np.zeros((n, n))
    companion_matrix[0, :] = -coeffs[1:]
    for i in range(1, n):
        companion_matrix[i, i - 1] = 1

    return companion_matrix


def lagrange(x, x_values, i):

    basis = 1.0
    epsilon = 1e-10

    for j in range(len(x_values)):
        if i != j:
            basis *= (x - x_values[j]) / ((x_values[i] - x_values[j]) + epsilon)
            
    return basis


def weights_for_legendre(roots):

    n = len(roots)
    weights = np.zeros(n)

    for i in range(n):
        weights[i] = quad(lagrange, -1, 1, args=(roots, i))[0]

    return weights


def signed_absolute(z_array):

    magnitudes = np.abs(z_array)
    signs = np.where(np.real(z_array) >= 0, 1, -1)

    return magnitudes * signs

n = 40
print(f"Legendre of Polynomial of order {n} :")
print(legendre_polynomial(n))
print()

companion_matrix = legendre_companion_matrix(n)
print("Companion Matrix :")
print(companion_matrix)
print()

eigen_values, eigen_vectors = np.linalg.eig(companion_matrix)
roots = np.sort(signed_absolute(eigen_values))
print(f"Roots of the Legendre Polynomial of order {n} :")
print(roots)
print()

weights = weights_for_legendre(roots)
print("Weights of the Roots :")
print(weights)
print()

print("Total Sum of Weights :")
print(np.sum(weights))
print()

plt.scatter(roots, weights, c="blue")
plt.xlabel("Roots")
plt.ylabel("Weights")
plt.title(f"Roots vs Weights plot for n = {n}")
plt.grid()
plt.show()