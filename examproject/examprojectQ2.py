import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import sympy as sm

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

