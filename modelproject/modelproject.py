import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from random import randrange

# Bertrand Model

def run_bertrand_model():
    # Parameters
    c = 10  # Marginal cost
    T = 109  # Number of periods
    a = 20  # Maximum price
    b = 0.5  # Elasticity
    price_change = 0.1  # Price change made by each firm after one period

    # Starting prices for each firm
    p1 = 15
    p2 = 18

    # Arrays to store prices and profits over time
    prices1 = np.zeros(T)
    prices2 = np.zeros(T)
    profits1 = np.zeros(T)
    profits2 = np.zeros(T)

    # The objective function that we want to minimize
    def objective(prices):
        p1, p2 = prices
        q1 = a - p1 * b
        q2 = a - p2 * b
        profit1 = (p1 - c) * q1
        profit2 = (p2 - c) * q2
        return -(profit1 + profit2)  # We take the negative since we want to minimize later

    # Starting guess
    initial_prices = np.array([p1, p2])

    # Minimize the objective function
    result = minimize(objective, initial_prices, method='SLSQP')

    # Retrieve the equilibrium prices
    equilibrium_prices = result.x

    # Reaction function for each firm
    def reaction(p1, p2):
        return (20 - p1) / 2, (20 - p2) / 2

    # Initialize equilibrium_period variable
    equilibrium_period = None

    # The repeated Bertrand game
    for t in range(T):
        q1, q2 = reaction(p1, p2)
        profits1[t] = (p1 - c) * q1
        profits2[t] = (p2 - c) * q2

        # Update the prices based on the price_change and the difference with marginal cost
        p1 -= price_change * (p1 - c)
        p2 -= price_change * (p2 - c)

        # Store the prices from each period
        prices1[t] = p1
        prices2[t] = p2

        # Check if prices equal the marginal cost
        if np.isclose(p1, c) and np.isclose(p2, c):
            equilibrium_period = t + 1
            break  # exits the simulation

    # Plot the prices and profits over time
    time_periods = np.arange(1, T + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(time_periods, prices1, label='Firm 1')
    plt.plot(time_periods, prices2, label='Firm 2')
    plt.axhline(c, color='red', linestyle='--', label='Marginal Cost')
    plt.xlabel('Time Period')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Bertrand Model: Prices over Time')

    plt.subplot(1, 2, 2)
    plt.plot(time_periods, profits1, label='Firm 1')
    plt.plot(time_periods, profits2, label='Firm 2')
    plt.xlabel('Time Period')
    plt.ylabel('Profit')
    plt.legend()
    plt.title('Bertrand Model: Profits over Time')

    plt.tight_layout()
    plt.show()

    print("Equilibrium Period:", equilibrium_period)


# Cournot Model

def run_cournot_model(q1, q2, c, a, T):
    # Set the parameters
    quantity_change = 0.1  # Quantity change after each period

    # Arrays to store quantities and profits over time
    quantities1 = np.zeros(T)
    quantities2 = np.zeros(T)
    profits1 = np.zeros(T)
    profits2 = np.zeros(T)

    # The objective function that we want to minimize
    def objective(quantities):
        q1, q2 = quantities
        p = a - q1 - q2  # Calculate the market price based on total quantity
        profit1 = (p - c) * q1
        profit2 = (p - c) * q2
        return -(profit1 + profit2)  # Negate the objective for maximization

    # Starting guesses
    initial_quantities = np.array([q1, q2])

    # We minimize the objective function
    result = minimize(objective, initial_quantities, method='SLSQP')

    # Retrieve the equilibrium quantities
    equilibrium_quantities = result.x

    # The repeated Cournot game
    for t in range(T):
        p = a - q1 - q2  # Calculate the market price based on total quantity
        profits1[t] = (p - c) * q1
        profits2[t] = (p - c) * q2
        
        # Update the quantities based on the quantity change and the difference with the optimal response
        q1 -= quantity_change * (q1 - equilibrium_quantities[0])
        q2 -= quantity_change * (q2 - equilibrium_quantities[1])
        
        # Store the current quantities
        quantities1[t] = q1
        quantities2[t] = q2

        # Print prices and quantities for each time period
        #print(f"Time Period {t+1}:")
        #print(f"Price: {p:.2f}")
        #print(f"Quantity - Firm 1: {q1:.2f}")
        #print(f"Quantity - Firm 2: {q2:.2f}")
        #print("")

    # Plot the quantities and profits over time
    time_periods = np.arange(1, T + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(time_periods, quantities1, label='Firm 1')
    plt.plot(time_periods, quantities2, label='Firm 2')
    plt.xlabel('Time Period')
    plt.ylabel('Quantity')
    plt.legend()
    plt.title('Cournot Model: Quantities over Time')

    plt.subplot(1, 2, 2)
    plt.plot(time_periods, profits1, label='Firm 1')
    plt.plot(time_periods, profits2, label='Firm 2')
    plt.xlabel('Time Period')
    plt.ylabel('Profit')
    plt.legend()
    plt.title('Cournot Model: Profits over Time')

    plt.tight_layout()
    plt.show()
