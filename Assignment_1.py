import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve
import pandas as pd

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
        # Make the diagonal element 1 by dividing the entire row by the diagonal element
        diag_element = augmented_matrix[i, i]
        if diag_element == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
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

df_1 = pd.read_excel("matrix.xlsx")
matrix = df_1.to_numpy()

print("Given matrix:")
print(matrix)
print()

#Eigenvalue calculation
eigenvalues = qr_algorithm(matrix)
print("Eigenvalues using LU decomposition:", eigenvalues)
print()

print("Determinant of the matrix equals the product of it's eigenvalues.")
det_matrix = np.linalg.det(matrix)



#Determinant calculation
det_eigen = 1
for eigenvalue in eigenvalues:
    det_eigen = det_eigen * eigenvalue

if det_matrix == 0:
    print("Determinant of the matrix is:", det_eigen)
    print("Thus the determinant can be approximated to 0")
    print("Either no or infinite solution as determinant is zero.")

else:
    print()
    print("Determinant of the matrix is:", det_eigen)
    print("Unique solution exists as determinant is not zero.")
    print()

    # Condition number calculation
    print("Condition number of a matrix is given by:")
    print("Eigen(max) / Eigen(min) = ", max(abs(eigenvalues)) / min(abs(eigenvalues)))
    print()

    k = max(abs(eigenvalues)) / min(abs(eigenvalues))
    print("Condition number of a 5x5 Hilbert Matrix is 476607.25024175597")

    if k > 476607.25024175597:
        print("The matrix is ill-conditioned.")
        print()

    else:
        print("The matrix is well behaved matrix")
        print()

        roots = eigenvalues
        # Create the polynomial coefficients from the roots
        coefficients = np.poly(roots)

        # Create a polynomial object
        polynomial = np.poly1d(coefficients)
        print("Polynomial function with these eigenvalues as their roots is:")
        print(polynomial)
        print()

        eigenvalue, eigenvector = power_method(matrix)
        print("Largest eigenvalue using Power Method:", eigenvalue)
        print()

        matrix_inverse = gauss_jordan_inverse(matrix)
        print("Inverse of the matrix using Gauss-Jordan Elimination:")
        print()
        print(matrix_inverse)
        print()

        eigenvalue, eigenvector = power_method(matrix_inverse)
        print("Largest eigenvalue of the inverse matrix using Power method:", eigenvalue)
        print("Thus the smallest eigenvalue for the initial matrix is:", 1 / eigenvalue)
        print()

        df_2 = pd.read_excel("ans1.xlsx")

        b1 = df_2.to_numpy()

        # Reshape the list into an n x n matrix
        ans1 = np.array(b1).reshape(5, 1)
        ans11 = np.array(b1).reshape(1, 5)
        print("Vector b1: ")
        print(ans11[0])
        print()

        df_3 = pd.read_excel("ans2.xlsx")

        b2 = df_3.to_numpy()

        # Reshape the list into an n x n matrix
        ans2 = np.array(b2).reshape(5, 1)
        ans22 = np.array(b2).reshape(1, 5)
        print("Vector b2: ")
        print(ans22[0])
        print()

        print("Ax = b")
        print("A = LU")
        print("LUx = b")
        print("Ux = y")
        print("Ly = b")
        print()

        P, L, U = lu(matrix)

        print("The L,U matrices for the given matrix is:")
        print()
        print("L: ")
        print()
        print(L) 
        print()
        print("U: ")
        print()
        print(U)
        print()

        X1, Y1 = solve_lu(P, L, U, b1)
        X2, Y2 = solve_lu(P, L, U, b2)

        print("Solution and the y vector for: ", ans11[0])
        print(X1.T[0])
        print("y: ", Y1.T[0])
        print()
        print("Solution and y for", ans22[0])
        print(X2.T[0])
        print("y: ", Y2.T[0])