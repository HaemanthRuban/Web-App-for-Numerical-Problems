import numpy as np
import matplotlib.pyplot as plt

def legendre_polynomial_coeffs(n):
    """Calculate the coefficients for the nth Legendre polynomial."""
    if n == 0:
        return [1]
    elif n == 1:
        return [1, 0]

    p_n_minus_2 = [1]
    p_n_minus_1 = [1, 0]

    for i in range(2, n + 1):
        a_n = 2 - (1 / i)
        c_n = 1 - (1 / i)

        # Use the recurrence relation to calculate P(i)
        p_n = [a_n * p for p in p_n_minus_1]
        p_n = [0] + p_n

        for j in range(len(p_n_minus_2)):
            p_n[j] -= c_n * p_n_minus_2[j]

        p_n_minus_2 = p_n_minus_1
        p_n_minus_1 = p_n

    return p_n

def compute_A_n(n):
    """Compute the value A(n) = sqrt(c(n+1) / (a(n) * a(n+1))) for the tri-diagonal matrix."""
    a_n = 2 - (1 / n)
    a_n_plus_1 = 2 - (1 / (n + 1))
    c_n_plus_1 = 1 - (1 / (n + 1))
    return np.sqrt(c_n_plus_1 / (a_n * a_n_plus_1))

def construct_legendre_matrix(n):
    """Construct the n x n symmetric tri-diagonal matrix for Legendre polynomial."""
    matrix = np.zeros((n, n))

    for i in range(n - 1):
        A_n = compute_A_n(i + 1)
        matrix[i, i + 1] = A_n
        matrix[i + 1, i] = A_n

    return matrix

def find_eigenvalues_and_unnormalized_vectors(matrix):
    """Compute eigenvalues and unnormalized eigenvectors of a matrix."""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    return eigenvalues, eigenvectors

def compute_weights(eigenvectors):
    """Compute Gaussian quadrature weights from eigenvectors for Legendre polynomials."""
    weights = (eigenvectors[0, :] ** 2)  # Square of the first element of each eigenvector
    # Scale to ensure the sum of weights is 2
    weights *= 2 / np.sum(weights)
    return weights

# Example usage
for n in range(2, 65):
    legendre_coeffs = legendre_polynomial_coeffs(n)
    print(f"P{n}:")

    legendre_matrix = construct_legendre_matrix(n)

    # Calculate eigenvalues and unnormalized eigenvectors
    eigenvalues, eigenvectors = find_eigenvalues_and_unnormalized_vectors(legendre_matrix)
    sorted_index = np.argsort(eigenvalues)
    roots = eigenvalues[sorted_index]
    print("Roots:")
    print(roots)  # Sorted for convenience

    # Calculate weights
    eigenvectors = eigenvectors[:, sorted_index]
    weights = compute_weights(eigenvectors)
    print("Weights:")
    print(weights)

    if n == 64:
        plt.scatter(roots, weights, c="blue")
        plt.xlabel("Roots")
        plt.ylabel("Weights")
        plt.title(f"Roots vs Weights plot for n = {n}")
        plt.grid()
        plt.show()

    


