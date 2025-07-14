import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp

# Title of the GUI
st.title("Generating Roots and Weights of Gauss-Legendre Polynomial")

# Option selection for the method
method = st.selectbox("Choose a method:", ["Using Jacobi Matrix", "Using Companion Matrix"])

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

# Option 1: Using Jacobi Matrix
if method == "Using Jacobi Matrix":
    order = st.slider("Select the order of the polynomial:", min_value=1, max_value=100, value=5)
    st.write("### Polynomial")
    coefficients = legendre_polynomial_coeffs(order)
    x = sp.symbols('x')
    legengre_polynomial = sum(c * x**i for i, c in enumerate(reversed(coefficients)))
    st.write(legengre_polynomial)

    n = order
    legendre_coeffs = legendre_polynomial_coeffs(n)
    legendre_matrix = construct_legendre_matrix(n)
    eigenvalues, eigenvectors = find_eigenvalues_and_unnormalized_vectors(legendre_matrix)
    sorted_index = np.argsort(eigenvalues)
    roots = eigenvalues[sorted_index] 
    eigenvectors = eigenvectors[:, sorted_index]
    weights = compute_weights(eigenvectors)

    # Display roots and weights in two columns with scrollable output
    st.write("### Roots and Weights")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Roots:")
        st.text_area("Roots", "\n".join(map(str, roots)), height=150)
    with col2:
        st.write("Weights:")
        st.text_area("Weights", "\n".join(map(str, weights)), height=150)

    # Plot roots vs weights
    st.write("### Plot of Roots vs Weights")
    fig, ax = plt.subplots()
    ax.scatter(roots, weights, color='blue')
    ax.set_xlabel("Roots")
    ax.set_ylabel("Weights")
    ax.set_title("Roots vs Weights Plot")
    ax.grid()
    st.pyplot(fig)

# Option 2: Using Companion Matrix
elif method == "Using Companion Matrix":
    st.write("(Note : Since the program uses numpy, the values are approximated so that the roots of the polynomial "+
                       "of order greater than 44 will be in complex and the weights will be some extreme values)")
    order = st.slider("Select the order of the polynomial:", min_value=1, max_value=100, value=5)
    st.write("### Polynomial")
    coefficients = legendre_polynomial_coeffs(order)
    x = sp.symbols('x')
    legengre_polynomial = sum(c * x**i for i, c in enumerate(reversed(coefficients)))
    st.write(legengre_polynomial)

    # Dummy companion matrix
    companion_matrix = legendre_companion_matrix(order)
    st.write("### Companion Matrix")
    st.dataframe(companion_matrix)

    n = order
    eigen_values, eigen_vectors = np.linalg.eig(companion_matrix)
    roots = np.sort(signed_absolute(eigen_values))
    weights = weights_for_legendre(roots)

    # Display roots and weights in two columns with scrollable output
    st.write("### Roots and Weights")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Roots:")
        st.text_area("Roots", "\n".join(map(str, roots)), height=150)
    with col2:
        st.write("Weights:")
        st.text_area("Weights", "\n".join(map(str, weights)), height=150)

    # Plot roots vs weights
    st.write("### Plot of Roots vs Weights")
    fig, ax = plt.subplots()
    ax.scatter(roots, weights, color='blue')
    ax.set_xlabel("Roots")
    ax.set_ylabel("Weights")
    ax.set_title("Roots vs Weights Plot")
    ax.grid()
    st.pyplot(fig)

