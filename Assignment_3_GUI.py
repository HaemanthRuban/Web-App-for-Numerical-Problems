import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def analytical_solution(y, P):
    u = -(P*(y**2) / 2) + (1 + P/2)*y
    return u

def thomas(a,b,c,d):
    """ A is the tridiagnonal coefficient matrix and d is the RHS matrix"""
    N = len(a)
    cp = np.zeros(N,dtype='float64') # store tranformed c or c'
    dp = np.zeros(N,dtype='float64') # store transformed d or d'
    X = np.zeros(N,dtype='float64') # store unknown coefficients
    
    # Perform Forward Sweep
    # Equation 1 indexed as 0 in python
    cp[0] = c[0]/b[0]  
    dp[0] = d[0]/b[0]
    # Equation 2, ..., N (indexed 1 - N-1 in Python)
    for i in np.arange(1,(N),1):
        dnum = b[i] - a[i]*cp[i-1]
        cp[i] = c[i]/dnum
        dp[i] = (d[i]-a[i]*dp[i-1])/dnum
    
    # Perform Back Substitution
    X[(N-1)] = dp[N-1]  # Obtain last xn 

    for i in np.arange((N-2),-1,-1):  # use x[i+1] to obtain x[i]
        X[i] = (dp[i]) - (cp[i])*(X[i+1])
    
    return(X)

def bvp_solution(delta_y, P, boundary_cond):

    y = y = np.arange(0, 1 + delta_y, delta_y)
    n = len(y)
    u_0, u_n = boundary_cond

    a = np.full(n, 1) # 1st lower diagonal
    a[0] = 0
    a[-1] = 0

    b = np.full(n, -2) # Diagonal elements
    b[0] = 1
    b[-1] = 1

    c = np.full(n, 1) # 1st upper diagonal
    c[0] = 0
    c[-1] = 0

    d = np.full(n, -P * delta_y**2) # RHS vector
    d[0] = u_0  # Boundary condition
    d[-1] = u_n # Boundary condition

    u = thomas(a, b, c, d)

    return y, u

def explicit_euler(h, P, initial):

    y = np.arange(0, 1 + h, h)
    n = len(y)
    u = np.zeros(n)
    u1 = np.zeros(n)
    u_0, u1_0 = initial
    u[0] = u_0
    u1[0] = u1_0

    for i in range(n-1):
        u1[i+1] = u1[i] - h * P
        u[i+1] = u[i] + h * u1[i]

    return y, u

def shoot_using_explicit(P):

    h = 0.02
    tol = 1e-6
    max_iters = 100
    low = -50
    high = 100
    count = 0

    while count <= max_iters:
        count = count + 1
        u_0 = 0
        u_n = 1
        u1_0 = np.mean([low, high])

        y, u = explicit_euler(h, P, (u_0, u1_0))
        
        if np.abs(u_n - u[-1]) <= tol:
            break

        if u_n - u[-1] < 0:
            high = u1_0
        else:
            low = u1_0

    return u1_0

def implicit_euler(h, P, initial):
    
    y = np.arange(0, 1 + h, h)
    n = len(y)
    u = np.zeros(n)
    u1 = np.zeros(n)
    u_0, u1_0 = initial
    u[0] = u_0
    u1[0] = u1_0

    for i in range(n-1):
        u1[i+1] = u1[i] - h * P
        u[i+1] = u[i] + h * u1[i+1]

    return y, u

def shoot_using_implicit(h, P):

    tol = 1e-6
    max_iters = 100
    low = -50
    high = 100
    count = 0

    while count <= max_iters:
        count = count + 1
        u_0 = 0
        u_n = 1
        u1_0 = np.mean([low, high])

        y, u = implicit_euler(h, P, (u_0, u1_0))
        
        if np.abs(u_n - u[-1]) <= tol:
            break

        if u_n - u[-1] < 0:
            high = u1_0
        else:
            low = u1_0

    return u1_0

# 1. Title
st.title("Couette Flow Problem")
st.write("The Couette flow probelem can be solved by many ways. But here we use different numerical methods "
         + "to solve it and compare it with the analytical solution to check the correctness of the methods we use.")
st.write("The problem given for us is second order ODE given by,")
st.latex(r'''
\frac{{d^2}u}{dy^2} = -P
''')
st.write("With boundary condtions, ")
st.latex(r'''
u(0) = 0 , u(1) = 1
         ''')

# 2. Add a slider for pressure (P)
st.sidebar.header("Input Parameters")
pressure = st.sidebar.slider("Pressure (P)", min_value=-50.0, max_value=100.0, value=10.0, step=1.0)
st.sidebar.write(f"Selected Pressure: {pressure}")

y = np.arange(0, 1.01, 0.01)
og_solution = analytical_solution(y, pressure)

# 3. Boundary Value Problem Section
st.subheader("Boundary Value Problem")
st.write("In this BVP, we use the finite difference formula, ")
st.latex(r'''
\frac{u_{i+1}- 2u_i + u_{i-1} }{(\Delta y)^2} = -P         
         ''')
st.write("Now if we write the above equation in matrix from we get,")
st.latex(r'''
\begin{bmatrix}
1 & 0 & 0 &  \ldots & 0 & 0 \\
1 & -2 & 1 &  \ldots & 0 & 0 \\
0 & 1 & -2 &  \ldots & 0 & 0 \\
\vdots &  \vdots & \vdots & \vdots & \vdots  & \vdots \\
0 & 0 & 0 &  \ldots & -2 & 1 \\
0 & 0 & 0 & \ldots & 0 & 1 \\
\end{bmatrix} 
\begin{bmatrix}
u_0\\
u_1 \\
u_2\\
\vdots\\
u_{99}\\
u_{100}\\
\end{bmatrix}  = 
\begin{bmatrix}
0\\
-(\Delta y)^2P \\
-(\Delta y)^2P \\
\vdots\\
-(\Delta y)^2P \\
1\\
\end{bmatrix}  
         ''')

st.write("Then we solve the matrix equation using **Thomas algorithm**")
bvp_step_size = st.number_input(
    "Step size for Boundary Value Problem", 
    min_value=0.01, max_value=0.5, value=0.1, step=0.01
)

y_bvp , u_bvp = bvp_solution(bvp_step_size, pressure, (0, 1))

if bvp_step_size > 0:
    # Placeholder for your plotting logic
    fig, ax = plt.subplots()
    ax.plot(y_bvp, u_bvp, 'or', label="Finite difference")  # Example plot
    ax.plot(y, og_solution, c="blue", linewidth=1.5, label="Analytical solution")
    ax.set_title("Boundary Value Problem Plot")
    ax.set_xlabel("Y-value")
    ax.set_ylabel("U-value")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

st.subheader("Converting BVP into IVP")
st.write("After converting it to IVP we get two ODEs as,")
st.latex(r'''
u^{'} = u_1 
         ''')
st.latex(r'''
u_1^{'} = -P 
         ''')
st.write("The initial condition is $u(0) = 0$ and the initial condition for $u_1$ " +
          "is given in such a way that it satisfies the boundary condition of $u$." +
" Let us say,  $u_1(0) = u^{'}(0) = s$,  such that $u(1) = 1$. To find $s$, we use **Shooting method**.")

# 4. Explicit Euler Section
st.subheader("Explicit Euler")
st.write("The formula of Explicit Euler is,")
st.latex(r'''
y_{n+1} = y_n + hy_n^{'}
         ''')
st.write("where, $h$ is the step size.")
st.write("If we use Explicit formual to two ODEs we get, ")
st.latex(r'''
u_{n+1} = u_n + hu_{1,n}
         ''')
st.latex(r'''
u_{1,n+1} = u_{1,n} - hP
         ''')

explicit_step_size = st.number_input(
    "Step size for Explicit Euler", 
    min_value=0.01, max_value=0.5, value=0.1, step=0.01
)

u1_0_explicit = shoot_using_explicit(pressure)
y_explicit , u_explicit = explicit_euler(explicit_step_size, pressure, (0, u1_0_explicit))

if explicit_step_size > 0:
    # Placeholder for your plotting logic
    fig, ax = plt.subplots()
    ax.plot(y_explicit, u_explicit, 'or', label="Explicit Euler")
    ax.plot(y, og_solution, c="blue", linewidth=1.5, label="Analytical solution")
    ax.set_title("Explicit Euler Plot")
    ax.set_xlabel("Y-value")
    ax.set_ylabel("U-value")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

# 5. Implicit Euler Section
st.subheader("Implicit Euler")

st.write("The formula of Implicit Euler is,")
st.latex(r'''
y_{n+1} = y_n + hy_{n+1}^{'}
         ''')
st.write("where, $h$ is the step size.")
st.write("If we use Implicit formual to two ODEs we get, ")
st.latex(r'''
u_{n+1} = u_n + hu_{1,n+1}
         ''')
st.latex(r'''
u_{1,n+1} = u_{1,n} - hP
         ''')

implicit_step_size = st.number_input(
    "Step size for Implicit Euler", 
    min_value=0.01, max_value=0.5, value=0.1, step=0.01
)

u1_0_implicit = shoot_using_implicit(implicit_step_size, pressure)
y_implicit , u_implicit = implicit_euler(implicit_step_size, pressure, (0, u1_0_implicit))

if implicit_step_size > 0:
    # Placeholder for your plotting logic
    fig, ax = plt.subplots()
    ax.plot(y_implicit, u_implicit, 'or', label="Implicit Euler")
    ax.plot(y, og_solution, c="blue", linewidth=1.5, label="Analytical solution")
    ax.set_title("Implicit Euler Plot")
    ax.set_xlabel("Y-value")
    ax.set_ylabel("U-value")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

