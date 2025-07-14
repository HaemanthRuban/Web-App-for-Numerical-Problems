import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
from scipy.linalg import lu

def solve_lu(P, L, U, b):
    # Step 1: Apply permutation to b -> inverse of(P) multiplied by b
    P = np.linalg.inv(P)
    Pb = np.dot(P, b)

    # Step 2: Forward substitution to solve Ly = Pb
    n = len(b)
    y = np.zeros_like(b, dtype=float)

    for i in range(n):
        y[i] = (Pb[i] - np.dot(L[i, :i], y[:i]))

    # Step 3: Backward substitution to solve Ux = y
    x = np.zeros_like(b, dtype=float)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x, y

def lu_decomposition_no_pivot(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    for i in range(n):
        # Upper Triangular Matrix U
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))

        # Lower Triangular Matrix L
        for k in range(i, n):
            if i == k:
                L[i, i] = 1  # Diagonal of L is 1
            else:
                L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[i, i]

    return L, U
def gauss_jordan_inverse(A):
    # Step 1: Augment the matrix A with the identity matrix I of the same size
    n = A.shape[0]
    augmented_matrix = np.hstack((A, np.eye(n)))

    # Step 2: Perform row operations to convert A to I
    for i in range(n):
        # Check if the diagonal element is zero
        if augmented_matrix[i, i] == 0:
            # Find a row below with a non-zero element in the same column
            for k in range(i + 1, n):
                if augmented_matrix[k, i] != 0:
                    # Swap the rows
                    augmented_matrix[[i, k]] = augmented_matrix[[k, i]]
                    break
            else:
                # If no suitable row is found, the matrix is singular
                raise ValueError("Matrix is singular and cannot be inverted.")

        # Make the diagonal element 1 by dividing the entire row by the diagonal element
        diag_element = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / diag_element

        # Make the other elements in the column 0 by subtracting suitable multiples of the row
        for j in range(n):
            if i != j:
                row_factor = augmented_matrix[j, i]
                augmented_matrix[j] = augmented_matrix[j] - row_factor * augmented_matrix[i]

    # Step 3: The right half of the augmented matrix is now the inverse of A
    A_inverse = augmented_matrix[:, n:]

    return A_inverse

def qr_algorithm(A, num_iterations=1000, tol=1e-10):
    A = np.array(A, dtype=float)
    n = A.shape[0]

    for i in range(num_iterations):
        Q, R = np.linalg.qr(A)
        A_next = R @ Q
        if np.allclose(A, A_next, atol=tol):
            Q_final, R_final = Q, R
            break
        A = A_next
    else:
        Q_final, R_final = Q, R  # In case the tolerance condition is not met within iterations
    return np.diag(A)

def power_method(A, num_iterations=1000, tol=1e-10):
    n = A.shape[0]
    # Step 1: Choose an initial vector (randomly)
    b = np.random.rand(n)

    # Normalize the initial vector
    b = b / np.linalg.norm(b)

    for _ in range(num_iterations):
        # Step 2: Multiply by the matrix
        b_new = np.dot(A, b)

        # Step 3: Normalize the resulting vector
        b_new_norm = np.linalg.norm(b_new)
        b_new = b_new / b_new_norm

        # Step 4: Compute the Rayleigh quotient (approximate eigenvalue)
        eigenvalue = np.dot(b_new.T, np.dot(A, b_new))

        # Check for convergence
        if np.linalg.norm(b_new - b) < tol:
            break

        b = b_new

    return eigenvalue, b_new

# 1. Title of the app
st.title("Matrix Operations GUI")

# 2. Introductory lines about the functioning of the GUI
st.write("This GUI allows you to upload a matrix and vectors, and perform various operations." + 
         "You can select different options for calculations like eigenvalues, determinant, condition number, and more.")

# 3. Uploading matrix file and displaying it
matrix_file = st.file_uploader("Upload a matrix file (Excel format)", type=["xlsx", "xls"])
vector_file_1 = st.file_uploader("Upload the first vector file (Excel format)", type=["xlsx", "xls"])
vector_file_2 = st.file_uploader("Upload the second vector file (Excel format)", type=["xlsx", "xls"])

if matrix_file:
    matrix_df = pd.read_excel(matrix_file)
    #matrix_df.index = [""] * matrix_df.shape[0]
    st.markdown("<h2 style='font-size:28px;'>Uploaded matrix</h2>", unsafe_allow_html=True)
    sympy_matrix = sp.Matrix(matrix_df.to_numpy())
    latex_matrix = sp.latex(sympy_matrix)
    st.latex(f'''{latex_matrix}''')

    if vector_file_1:
        st.markdown("<h2 style='font-size:28px;'>Uploaded vectors</h2>", unsafe_allow_html=True)
        vector_df_1 = pd.read_excel(vector_file_1)
        st.write("Uploaded Vector 1:")
        sympy_matrix_1 = sp.Matrix(vector_df_1.to_numpy())
        latex_matrix_1 = sp.latex(sympy_matrix_1)
        st.latex(f'''{latex_matrix_1}''')

    if vector_file_2:
        vector_df_2 = pd.read_excel(vector_file_2)
        st.write("Uploaded Vector 2:")
        sympy_matrix_2 = sp.Matrix(vector_df_2.to_numpy())
        latex_matrix_2 = sp.latex(sympy_matrix_2)
        st.latex(f'''{latex_matrix_2}''')


    # Convert uploaded matrix to a NumPy array (using dummy array if no file uploaded)
    if matrix_file:
        matrix = matrix_df.to_numpy()

    #Eigenvalue calculation
    eigenvalues = qr_algorithm(matrix)

    # Determinant
    det_eigen = np.prod(eigenvalues)

    # 4. Multiselect for matrix operations
    options = st.multiselect(
        "Select operations to perform:",
        [
            "Calculate Eigenvalues",
            "Calculate Determinant",
            "Calculate Condition Number",
            "Calculate Polynomial",
            "Highest Eigenvalue by Power Method",
            "Inverse Matrix",
            "Smallest Eigenvalue by power method" ,
            "Find solutions"
        ]
    )

    # Display dummy results based on selection
    if "Calculate Eigenvalues" in options:
        st.markdown("<h2 style='font-size:28px;'>Eigenvalues</h2>", unsafe_allow_html=True)
        st.dataframe(eigenvalues.T)

    if "Calculate Determinant" in options:
        st.markdown("<h2 style='font-size:28px;'>Determinant</h2>", unsafe_allow_html=True)
        if np.abs(det_eigen) < 1e-8:
            st.write("**Determinant** = 0")
            st.write("There is no unique solution since determinant is zero")
        else:
            st.write(f"Determinant $= {det_eigen}$")
            st.write("There is unique solution since determinant is not equal to zero")
        

    if "Calculate Condition Number" in options:
        st.markdown("<h2 style='font-size:28px;'>Condition number</h2>", unsafe_allow_html=True)
        k = max(abs(eigenvalues)) / min(abs(eigenvalues))
        st.write(f"Condition Number of given matrix     $ =  {k}$")
        st.write("Condition Number of Hilbert matrix is $=  476607.25024175597$")

        if k > 476607.25024175597:
            st.write("The matrix is ill-conditioned.")
        else:
            st.write("The matrix is well behaved matrix")

    if "Calculate Polynomial" in options:
        st.markdown("<h2 style='font-size:28px;'>Polynomial whose roots are eigenvalues</h2>", unsafe_allow_html=True)
        roots = eigenvalues
        coefficients = np.poly(roots)
        x = sp.symbols('x')
        polynomial = sum(c * x**i for i, c in enumerate(reversed(coefficients)))
        st.write(polynomial)

    if "Highest Eigenvalue by Power Method" in options:
        st.markdown("<h2 style='font-size:28px;'>Highest Eigenvalue by Power Method</h2>", unsafe_allow_html=True)
        high_eigenvalue, eigenvector = power_method(matrix)
        st.write(f"${high_eigenvalue}$")

    if "Inverse Matrix" in options:
        st.markdown("<h2 style='font-size:28px;'>Inverse of the matrix</h2>", unsafe_allow_html=True)
        if np.abs(det_eigen) < 1e-8:
            st.write("There is no inverse since determinant is zero")
        else:
            matrix_inverse = gauss_jordan_inverse(matrix)
            matrix_inverse = np.around(matrix_inverse, decimals=2)
            sympy_matrix_3 = sp.Matrix(matrix_inverse)
            latex_matrix_3 = sp.latex(sympy_matrix_3)
            st.latex(f'''{latex_matrix_3}''')

    if "Smallest Eigenvalue by power method" in options:
        st.markdown("<h2 style='font-size:28px;'>Lowest Eigenvalue by Power Method</h2>", unsafe_allow_html=True)
        if np.abs(det_eigen) < 1e-8:
            st.write("Cannot find using power method since inverse is not defined because determinant is zero")
        else:
            eigenvalue, eigenvector = power_method(matrix_inverse)
            low_eigenvalue = 1 / eigenvalue
            st.write(f"${low_eigenvalue}$")

    if "Find solutions" in options:

        # Convert uploaded vectors to NumPy arrays (using dummy arrays if no file uploaded)
        vector_1 = vector_df_1.to_numpy() 
        vector_2 = vector_df_2.to_numpy()

        determinant = det_eigen
        k = max(abs(eigenvalues)) / min(abs(eigenvalues))
        condition_number = k  
        hilbert_condition_number = 476607.25024175597

        st.markdown("<h2 style='font-size:28px;'>Solution using LU decomposition</h2>", unsafe_allow_html=True)

        if np.abs(determinant) > 1e-6 and condition_number < hilbert_condition_number:
            P, L, U = lu(matrix)
            X1, Y1 = solve_lu(P, L, U, vector_1)
            X2, Y2 = solve_lu(P, L, U, vector_2)
            X1 = np.around(X1, decimals=2)
            X2 = np.around(X2, decimals=2)
            st.write("Solution Vector for Vector 1: ")
            sympy_matrix_4 = sp.Matrix(X1)
            latex_matrix_4 = sp.latex(sympy_matrix_4)
            st.latex(f'''{latex_matrix_4}''')
            st.write("Solution Vector for Vector 2: ")
            sympy_matrix_5 = sp.Matrix(X2)
            latex_matrix_5 = sp.latex(sympy_matrix_5)
            st.latex(f'''{latex_matrix_5}''')
        else:
            st.write("Cannot solve the system: Determinant is zero or condition number is too high.")