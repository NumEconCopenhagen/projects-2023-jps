import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import sympy as sm

# QUESTION 1 

# 1.1

# Define parameters
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

# Question 2

# 2.1  

def calculate_optimal_ell():
    # Define the symbols
    kappa = sm.symbols('kappa_t')
    eta = sm.symbols('eta')
    w = sm.symbols('w')
    ell = sm.symbols('ell_t')

    # Define the profit function
    profit = kappa * ell**(1 - eta) - w * ell

    # Take the derivative of the profit function with respect to ell_t
    FOC = sm.diff(profit, ell)

    # Solve the first order condition for ell_t, i.e., set it to zero and solve for ell_t
    optimal_ell = sm.solve(FOC, ell)

    # Display the solution
    print("The optimal ell_t = ", optimal_ell)

def calculate_optimal_profit_num():
   # Define the symbols
    kappa = sm.symbols('kappa_t')
    eta = sm.symbols('eta')
    w = sm.symbols('w')
    ell = sm.symbols('ell_t')

    # Define the profit function
    profit = kappa * ell**(1 - eta) - w * ell

    # Take the derivative of the profit function with respect to ell_t
    FOC = sm.diff(profit, ell)

    # Solve the first order condition for ell_t, i.e., set it to zero and solve for ell_t
    optimal_ell = sm.solve(FOC, ell)

    # Substitute the baseline parameters
   
    eta_value = 0.5
    w_value = 1.0

    # Substitute eta and w values into the optimal labor supply we use [0] to extract the first element of the list which is the solution to to FOC
    optimal_ell_value = [optimal_ell[0].subs({eta: eta_value, w: w_value, kappa: value}) for value in [1.0, 2.0]]

    profit1 = profit.subs({eta: eta_value, w: w_value, kappa: 1.0, ell: optimal_ell_value[0]})
    profit2 = profit.subs({eta: eta_value, w: w_value, kappa: 2.0, ell: optimal_ell_value[1]})

    # Display the solution with substituted values
    print(f"The optimal hairdressers for kappa = 1 and 2: {optimal_ell_value}")
    print(f"Profit for kappa = 1: {profit1}")
    print(f"Profit for kappa = 2 {profit2}")

# 2.2 
# Define the parameters
eta = 0.5
w = 1.0
rho = 0.90
iota = 0.01
sigma_epsilon = 0.10
R = (1 + 0.01)**(1/12)
T = 120
K = 10000  # Number of random shock series

def calculate_profit():
    # Define the policy function
    def policy(kappa):
        return ((1 - eta) * kappa / w)**(1 / eta)

    # Define the profit function
    def profit(kappa, ell, ell_prev):
        return kappa * ell**(1 - eta) - w * ell - (ell != ell_prev) * iota

    # Creating an array to store the ex values
    h_values = np.zeros(K)

    # Generate K random shock series and calculate the ex post value for each
    np.random.seed(100)  # Set a random seed for reproducibility
    for k in range(K): # For each shock series
        epsilon = np.random.normal(-0.5 * sigma_epsilon**2, sigma_epsilon, T) # we draw from a normal distribution with mean -0.5 * sigma_epsilon**2 and standard deviation sigma_epsilon
        kappa = np.exp(rho * np.log(1) + epsilon) # we are taking the exponential since kappa is log transformed in the description
        ell_prev = 0  
        for t in range(T): # For each period
            ell = policy(kappa[t])
            h_values[k] += R**(-t) * profit(kappa[t], ell, ell_prev) # from the description we know that the discount factor is R^(-t)
            ell_prev = ell

    # Calculate the expected value
    H = h_values.mean()
    print(f"The expected profit is {H:.2f}")


# 2.3 
def calculate_new_profit():
    # New policy function
    def policy(kappa, ell_prev, Delta):
        ell_star = ((1 - eta) * kappa / w) ** (1 / eta)
        if abs(ell_prev - ell_star) > Delta:
            return ell_star
        else:
            return ell_prev

   # Define the profit function
    def profit(kappa, ell, ell_prev):
        return kappa * ell**(1 - eta) - w * ell - (ell != ell_prev) * iota
    # Creating an array to store the ex values
    h_values_new = np.zeros(K)

    # Generate K random shock series and calculate the ex values
    np.random.seed(100)  
    Delta = 0.05  # Adjustment set
    for k in range(K): # For each shock series
        epsilon = np.random.normal(-0.5 * sigma_epsilon**2, sigma_epsilon, T)
        kappa = np.exp(rho * np.log(1) + epsilon)  # Assume kappa = 1
        ell_prev = 0  # Assume ell = 0
        for t in range(T): # For each period
            ell = policy(kappa[t], ell_prev, Delta)
            h_values_new[k] += R**(-t) * profit(kappa[t], ell, ell_prev)
            ell_prev = ell
    # Calculate the ex expected value for the new policy
    H_new = np.mean(h_values_new)
    return H_new

# 2.4 

# 2.5 
# We Change the discount rate for future profits 
R1 = (1 + 0.001)**(1/12)
def calculate_expected_profit_discount_rate():
    # New policy function
    def policy(kappa, ell_prev, Delta):
        ell_star = ((1 - eta) * kappa / w) ** (1 / eta)
        if abs(ell_prev - ell_star) > Delta:
            return ell_star
        else:
            return ell_prev
    
    # Define the profit function
    def profit(kappa, ell, ell_prev):
        return kappa * ell**(1 - eta) - w * ell - (ell != ell_prev) * iota
    
    # Creating an array to store the ex values
    h_values_new = np.zeros(K)
    
    # Generate K random shock series and calculate the ex post value for each
    np.random.seed(100)  # Set a random seed for reproducibility
    Delta = 0.05  # Adjustment threshold
    for k in range(K): # For each shock series
        epsilon = np.random.normal(-0.5 * sigma_epsilon**2, sigma_epsilon, T)
        kappa = np.exp(rho * np.log(1) + epsilon)  # Assume kappa_{-1} = 1
        ell_prev = 0  # Assume ell_{-1} = 0
        for t in range(T): # For each period
            ell = policy(kappa[t], ell_prev, Delta)
            h_values_new[k] += R1**(-t) * profit(kappa[t], ell, ell_prev)
            ell_prev = ell
    
    # Calculate the ex ante expected value for the new policy
    H_new = np.mean(h_values_new)
    
    # Calculate the difference in expected profits
    diff = H_new - H
    
    # Return the results
    return H, H_new, diff

# Call the function to get the results
old_profit, new_profit, diff = calculate_expected_profit_discount_rate()

# Display the results
print(f"Old expected profit: {old_profit:.2f}")
print(f"New expected profit: {new_profit:.2f}")
print(f"The difference is {diff:.2f}")


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