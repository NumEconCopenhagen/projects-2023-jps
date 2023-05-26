import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import sympy as sm

# Problem 2

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

    print(f"New expected profit: {H_new:.2f}")

# 2.4 

def calculate_optimal_delta():
    # Function to calculate ell_t given kappa_t
    def calculate_ell(kappa):
        return ((1 - eta) * kappa / w) ** (1 / eta)

    # Number of shock series to average over
    K = 100

    # Generate shock series
    np.random.seed(100)
    shock_series = np.random.normal(-0.5 * sigma_epsilon ** 2, sigma_epsilon, size=(K, T))

    # Range of delta values to search over
    delta_max = np.linspace(0, 0.1, 100)

    # Calculate ex-post values for each delta value
    values_prev = np.zeros(len(delta_max))

    for d, delta in enumerate(delta_max):
        delta_prev = np.zeros(K)

        for k in range(K):
            kappa_minus_1 = 1.0  # Initial shock
            ell_minus_1 = 0.0  # Initial employment

            value_prev = 0.0

            for t in range(T):
                kappa = np.exp(rho * np.log(kappa_minus_1) + shock_series[k, t])
                ell_prev = calculate_ell(kappa)

                if np.abs(ell_minus_1 - ell_prev) > delta:
                    ell = ell_prev
                else:
                    ell = ell_minus_1

                profit = kappa * ell ** (1 - eta) - w * ell - (ell != ell_minus_1) * iota
                value_prev += R ** (-t) * profit

                kappa_minus_1 = kappa
                ell_minus_1 = ell

            delta_prev[k] = value_prev

        values_prev[d] = np.mean(delta_prev)

    # Find optimal delta that maximizes H
    optimal_delta = delta_max[np.argmax(values_prev)]
    H_max = np.max(values_prev)
    print(f"Optimal delta: {optimal_delta:.3f}")
    print(f"Maximum expected profit: {H_max:.3f}")

     # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(delta_max, values_prev, label='Profit')
    plt.scatter(optimal_delta, H_max, color='red') 
    plt.xlabel('$\Delta$')
    plt.ylabel('Profit')
    plt.title('Optimal $\Delta$ for profit maximization')
    plt.legend()
    plt.grid(True)
    plt.show()


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
    print(f"New expected profit: {H_new:.2f}")
    

