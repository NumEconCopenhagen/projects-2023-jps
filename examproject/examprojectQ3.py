import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import sympy as sm

# Question 3

# 3.1 and 3.2

def griewank(x):
    return griewank_(x[0],x[1])
    
def griewank_(x1,x2):
    A = x1**2/4000 + x2**2/4000
    B = np.cos(x1/np.sqrt(1))*np.cos(x2/np.sqrt(2))
    return A-B+1

def griewank_solve(K_bar):
    min_ = -600
    max_ = 600
    tolerance = 10e-08
    K = 1000

    xstar = np.array([0, 0])
    xk = np.random.uniform(min_, max_, (K,2))

    for k in range(K):
        if griewank(xstar) >= tolerance or k == 0:
            if k >= K_bar:
                Chi = 1/(1+np.exp((k-K_bar)/100))
                xk[k] = Chi*xk[k]+(1-Chi)*xstar
            xkstar_ = minimize(griewank, xk[k], method='BFGS', tol=tolerance)
            xkstar = xkstar_.x
            if k == 0 or griewank(xkstar) < griewank(xstar):
                xstar = xkstar
            if griewank(xstar) < tolerance:
                kstar = k

    xk = np.delete(xk, slice(kstar, K), 0)

    print(f'x*=(x1*,x2*)=({round(xstar[0],10)}, {round(xstar[1],10)})')
    print(f'Last period = {len(xk)}')

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1)  
    ax1.plot(np.arange(kstar), np.delete(xk, 1, 1), label='$x_1$', marker='.', linestyle='None')
    ax1.plot(np.arange(kstar), np.delete(xk, 0, 1), label='$x_2$', marker='.', linestyle='None')
    ax1.legend(frameon=True, fontsize='large')
    ax1.set_xlabel('k')
    ax1.set_ylabel('$x^{k0}$')
    ax1.set_title('Griewank function in 2D')
    ax1.grid(True)
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')  
    ax2.scatter(np.delete(xk, 1, 1), np.delete(xk, 0, 1), np.arange(kstar))
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('Griewank function in 3D')

    plt.tight_layout()
    plt.show()