import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from random import randrange
import sympy as sm





# Bertrand Model
def run_bertrand_model(p1, p2, c, a):
    # Parameters
    b = 0.8  # Elasticity
    price_change = 0.05  # Price change made by each firm after one period

    # Arrays to store prices and profits over time
    prices1 = np.empty((0), float)
    prices2 = np.empty((0), float)
    profits1 = np.empty((0), float)
    profits2 = np.empty((0), float)

    prices1 = np.append(prices1, [p1], axis=0)
    prices2 = np.append(prices2, [p2], axis=0)

    if p1 < p2:
        profits1 = np.append(profits1, [(a-p1*b)*(p1-c)], axis=0)
        profits2 = np.append(profits2, [0], axis=0)
    if p1 == p2:
        profits1 = np.append(profits1, [(a-p1*b)*(p1-c)/2], axis=0)
        profits2 = np.append(profits2, [(a-p2*b)*(p2-c)/2], axis=0)
    if p1 > p2:
        profits1 = np.append(profits1, [0], axis=0)
        profits2 = np.append(profits2, [(a-p2*b)*(p2-c)], axis=0)

    while p1 > c or p2 > c:
        if p1 < p2:
            p1_ = p1
            p2_ = max(min((p1-c)*(1-price_change)+c,p1-0.01),c)
        if p1 > p2:
            p1_ = max(min((p2-c)*(1-price_change)+c,p2-0.01),c)
            p2_ = p2
        if p1 == p2:
            p1_ = max(min((p2-c)*(1-price_change)+c,p2-0.01),c)
            p2_ = max(min((p1-c)*(1-price_change)+c,p1-0.01),c)
        p1 = p1_
        p2 = p2_
        prices1 = np.append(prices1, [p1], axis=0)
        prices2 = np.append(prices2, [p2], axis=0)

        if p1 < p2:
            profits1 = np.append(profits1, [(a-p1*b)*(p1-c)], axis=0)
            profits2 = np.append(profits2, [0], axis=0)
        if p1 == p2:
            profits1 = np.append(profits1, [(a-p1*b)*(p1-c)/2], axis=0)
            profits2 = np.append(profits2, [(a-p2*b)*(p2-c)/2], axis=0)
        if p1 > p2:
            profits1 = np.append(profits1, [0], axis=0)
            profits2 = np.append(profits2, [(a-p2*b)*(p2-c)], axis=0)

    # Plot the prices and profits over time
    time_periods = np.arange(0, np.shape(prices1)[0])

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

# Cournot Model

def run_cournot_model(q1, q2, c, a, T):

    # Arrays to store quantities and profits over time
    quantities1 = np.zeros(T)
    quantities2 = np.zeros(T)
    profits1 = np.zeros(T)
    profits2 = np.zeros(T)

    # The objective function that we want to minimize
    def objective1(q1):
        p = a - q1 - q2  # Calculate the market price based on total quantity
        profit1 = (p - c) * q1
        return -(profit1)  # Negate the objective for maximization
    
    def objective2(q2):
        p = a - q1 - q2  # Calculate the market price based on total quantity
        profit2 = (p - c) * q2
        return -(profit2)  # Negate the objective for maximization

    # The repeated Cournot game
    for t in range(T):
        q1_ = minimize(objective1, q1, method='SLSQP')
        q2_ = minimize(objective2, q2, method='SLSQP')
        q1 = q1_.x
        q2 = q2_.x
        quantities1[t] = q1
        quantities2[t] = q2
        p = a - q1 - q2  # Calculate the market price based on total quantity
        profits1[t] = (p - c) * q1
        profits2[t] = (p - c) * q2

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

    print(f"Price: {a - q1 - q2}")
    print(f"Quantity - Firm 1: {round(quantities1[T-1],2)}")
    print(f"Quantity - Firm 2: {round(quantities2[T-1],2)}")
    print(f"Profit - Firm 1: {round(profits1[T-1],2)}")
    print(f"Profit - Firm 2: {round(profits2[T-1],2)}")