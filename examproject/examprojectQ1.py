import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import sympy as sm

# Problem 1 

# 1.1

# We define the parameters
alpha = 0.5
kappa = 1.0
nu = 1 / (2 * 16**2)

def L_star(w, tau, alpha, kappa, nu): #Define optimal labour supply
    tilde_w = (1 - tau) * w
    return (-kappa + np.sqrt(kappa**2 + 4 * alpha / nu * tilde_w**2)) / (2 * tilde_w)

def V(L, G, tau, alpha, kappa, w, nu): # Define utility
    C = kappa + (1 - tau) * w * L
    return np.log(C**alpha * G**(1 - alpha)) - nu * L**2 / 2

def solve_L(w, tau):

    def objective(L): # function to be minimized
        return -V(L, G, tau, alpha, kappa, w, nu)

    for G in [1.0, 2.0]: # Maximize utility for the given values of G
        L_ = minimize(objective, 12, method='Nelder-Mead')
        L = L_.x
        print(f"For G = {G}, optimal labor supply = {L}")
    
    Lstar = L_star(w, tau, alpha, kappa, nu)
    
    print(f'L* = {Lstar}')

# 1.2

# Set up a range of w values
def plot_Q2():
    w_values = np.linspace(0.01, 2, 100)  # We start from 0.01 to avoid division by zero in L_star

    # Calculate corresponding L_star values
    L_star_values = [L_star(w, 0.3, alpha, kappa, nu) for w in w_values]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(w_values, L_star_values, label='$L^{\star}(\\tilde{w})$')
    plt.xlabel('w', fontsize=14)
    plt.ylabel('$L^{\star}$', fontsize=14)
    plt.title('Optimal labor supply as a function of w', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.style.use('seaborn-whitegrid')
    plt.show()

# 1.3 

def plot_on_tau(w):
    # we define a function that returns the utility, L and G for a given tau
    tau_values = np.linspace(0.0, 1.0, 100)
    L_values = []
    G_values = []
    V_values = np.array([0.0])

    # we loop over the tau values and calculate the corresponding L, G and utility
    for tau in tau_values:
        L = L_star(w, tau, alpha, kappa, nu)
        G = tau * w * L
        utility = V(L, G, tau, alpha, kappa, w, nu)
        L_values.append(L)
        G_values.append(G)
        if tau == 0:
            V_values[0] = utility
        if tau != 0:
            V_values = np.append(V_values, utility)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(tau_values, L_values)
    plt.xlabel('$\\tau$')
    plt.ylabel('$L$')
    plt.title('Optimal Labor Supply')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(tau_values, G_values, color='green')
    plt.xlabel('$\\tau$')
    plt.ylabel('$G$')
    plt.title('Government Consumption')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(tau_values, V_values, color='red')
    plt.xlabel('$\\tau$')
    plt.ylabel('Utility')
    plt.title('Worker Utility')
    plt.grid(True)
    plt.style.use('seaborn-whitegrid')

    plt.tight_layout()
    plt.show()

# 1.4 

def max_of_tau(w, plot_fig):

    # we define a function that returns the utility, L and G for a given tau
    tau_values = np.linspace(0.0, 1.0, 100)
    L_values = []
    G_values = []
    V_values = np.array([0.0])

    # we loop over the tau values and calculate the corresponding L, G and utility
    for tau in tau_values:
        L = L_star(w, tau, alpha, kappa, nu)
        G = tau * w * L
        utility = V(L, G, tau, alpha, kappa, w, nu)
        L_values.append(L)
        G_values.append(G)
        if tau == 0:
            V_values[0] = utility
        if tau != 0:
            V_values = np.append(V_values, utility)

    max_utility = max(V_values)
    max_tau = np.where(V_values == max(V_values))[0][0]/100

    if plot_fig == 1:

        print(f'Optimal tax rate: {max_tau}')

        plt.subplot(1, 1, 1)
        plt.plot(tau_values, V_values)
        plt.plot(max_tau, max_utility, 'ro')
        plt.xlabel('$\\tau$')
        plt.ylabel('Utility')
        plt.title('Worker Utility')
        plt.grid(True)
        plt.style.use('seaborn-whitegrid')

    return max_tau

# 1.5

#Define new utilityfunction
def V_new(L, G, tau, alpha, kappa, w, nu, sigma, rho, epsilon):
    C = kappa + (1 - tau) * w * L
    Bracket = (alpha * C**((sigma - 1) / sigma) + (1 - alpha) * G**((sigma - 1) / sigma))**(sigma / (sigma - 1))
    return ((Bracket**(1 - rho) - 1) / (1 - rho) - nu * L**(1 + epsilon) / (1 + epsilon))

def solve_G(w, tau, sigma, rho, epsilon):

    def objective(L, G, tau, alpha, kappa, w, nu, sigma, rho, epsilon):
        return -V_new(L, G, tau, alpha, kappa, w, nu, sigma, rho, epsilon)

    # Function to find the optimal labor for given G and tau with a start ques of 12
    def solve_worker(G, tau, sigma, rho, epsilon):
        result = minimize(objective, [12], args=(G, tau, alpha, kappa, w, nu, sigma, rho, epsilon), bounds=[(0, 24)], method='SLSQP')
        return result.x[0]
    
    # Function to find G that solves the government problem
    def solve_government(tau, sigma, rho, epsilon):
        G = 0.5  # initial guess
        for _ in range(100):  # maximum of 100 iterations
            Lstar = solve_worker(G, tau, sigma, rho, epsilon)
            new_G = tau * w * Lstar
            if abs(new_G - G) < 1e-6:  # break if converged
                break
            G = new_G
        return G
    
    return(solve_government(tau, sigma, rho, epsilon))

# 1.6

def max_of_tau_new(w, sigma, rho, epsilon):

    def objective(L, G, tau, alpha, kappa, w, nu, sigma, rho, epsilon):
        return -V_new(L, G, tau, alpha, kappa, w, nu, sigma, rho, epsilon)

    # Function to find the optimal labor for given G and tau with a start ques of 12
    def solve_worker(G, tau, sigma, rho, epsilon):
        result = minimize(objective, 12, args=(G, tau, alpha, kappa, w, nu, sigma, rho, epsilon), bounds=[(0, 24)], method='SLSQP')
        return result.x[0]   

    def objective2(tau, w, sigma, rho, epsilon):
        G = solve_G(w, tau, sigma, rho, epsilon)
        L = solve_worker(G, tau, sigma, rho, epsilon)
        return -V_new(L, G, tau, alpha, kappa, w, nu, sigma, rho, epsilon)
    
    tau_star_ = minimize(objective2, 0.5, args=(w, sigma, rho, epsilon), method='Nelder-Mead', bounds=[(0, 1)])
    tau_star = tau_star_.x[0]
    return tau_star