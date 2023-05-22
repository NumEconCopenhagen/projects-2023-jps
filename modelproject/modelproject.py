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

<<<<<<< HEAD
=======
    print("Equilibrium Period:", equilibrium_period)
    firm_1_demand = sm.Eq(q1, 13 - 0.5*p1 + p2)
    firm_2_demand = sm.Eq(q1,  18 + p1 - 0.5*p2)

    print('Demand functions')

    display(firm_1_demand)
    display(firm_2_demand)
    # We define the variables that we will use later on
    p1, p2 = sm.symbols('p1 p2', real=True)
    qne = sm.symbols('Q_NE', real=True)
    pne = sm.symbols('P_NE', real = True)
    q1,q2 = sm.symbols('q1 q2', real=True)
    P1, P2 = sm.symbols('\Pi_1 \Pi_2', real=True)
    c = sm.symbols('c', real=True)
    # Then we calculate the profit functions from the 2 firms
    firm_1_profit = sm.Eq(P1, p1*(13 - 0.5*p1 + p2)-c*(13 - 0.5*p1 + p2))
    firm_2_profit = sm.Eq(P2, p2*(18 + p1 -0.5*p2)-c*(18 + p1 - 0.5*p2))

    print('Profit functions')

    display(firm_1_profit)
    display(firm_2_profit)

    # Now we calculate the first order conditions for the 2 firms and set equal to zero
    foc_1 = sm.diff(p1*(13 - 0.5*p1 + p2)-c*(13 - 0.5*p1 + p2),p1)
    foc_2 = sm.diff(p2*(18 + p1 - 0.5*p2)-c*(18 + p1 - 0.5*p2),p2)

    print('First order conditions')

    display(sm.Eq(FOC_1,0))
    display(sm.Eq(FOC_2,0))

    # Now we calculate the 2 firms reaction functions
    reac1 = sm.solve(FOC_1,p1)
    reac2 = sm.solve(FOC_2,p2)

    reac1_1 = sm.Eq(reac1[0],p1)
    reac2_2 = sm.Eq(reac2[0],p2)

    print('Reaction functions') 

    display(reac1_1)
    display(reac2_2)

    # now we isolate p2 in firm 1 reaction function, so that we can set them equal to each other
    reac1_p2 = sm.solve(reac1_1, p2)
    reac11_p2 = sm.Eq(reac1_p2[0],p2)

    print('P2 isolated in firm 1 reaction')
    display(reac11_p2)

    # We set the reaftion functions equal to each other
    equal = sm.Eq(reac2[0],reac1_p2[0])

    print('Reaction functions from firm 1 and 2 is set to equal')

    display(equal)

    # We calculate the the nash equilibrium price 
    nep = c + 31 

    nep1 = sm.Eq(pne,nep)

    print('Nash equilibrium price')

    display(nep1)

    # We insert the price in the demand function and then we get the Nash equilibrium quantity
    neq = sm.Eq(qne, 13 - 0.5*nep + nep)

    print('Nash quilibrium quantity')

    display(neq)

>>>>>>> bb27e965e8808f14bc0f8b23d39199a436c6240f
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