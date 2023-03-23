
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt


class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 0:
            H = np.fmin(HM,HF)
        elif par.sigma == 1:
            H = (HM+1e-10)**(1-par.alpha+1e-10)*(HF+1e-10)**par.alpha+1e-10
        else:
            H = ((1-par.alpha+1e-10)*(HM+1e-10)**((par.sigma-1)/(par.sigma+1e-10))+(par.alpha+1e-10)*(HF+1e-10)**((par.sigma-1)/(par.sigma+1e-10)))**((par.sigma+1e-10)/(par.sigma-1+1e-10))
        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho+1e-10)/(1-par.rho+1e-10)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """
        obj = lambda x: - self.calc_utility(x[0], x[1], x[2], x[3])    
        bounds = [(0,24)]*4
        guess = [4]*4
        result = optimize.minimize(obj, guess, method='Nelder-Mead',bounds=bounds)
        opt = SimpleNamespace()
        opt.LM = result.x[0]
        opt.HM = result.x[1]
        opt.LF = result.x[2]
        opt.HF = result.x[3]
        opt.util = self.calc_utility(opt.LM, opt.HM, opt.LF, opt.HF)
        
        return opt
    
    def estimate(self,alpha=None,sigma=None):
        wF_values = [0.8, 0.9, 1.0, 1.1, 1.2]
        par = self.par
        sol=self.sol
        def objective(x, self):                              #Define the function to be optimized
            par.alpha = x[0]
            par.sigma = x[1]
            for i, wF in enumerate(wF_values):                 #Solving the model for every female wage in the vector
                par.wF = wF
                out = self.solve()
                sol.LM_vec[i] = out.LM
                sol.LF_vec[i] = out.LF
                sol.HM_vec[i] = out.HM
                sol.HF_vec[i] = out.HF
            x = np.log(wF_values)                                #Saving the log wF - log wH vector (wH is normalized to one)
            y = np.log(sol.HF_vec/(sol.HM_vec+1e-10)) #Saving the log HF - log HM vector
            A = np.vstack([x, np.ones(len(x))]).T #Creating the matrix A, that lstsq uses as input. y=A*beta, where y and beta are vectors
            sol.beta1, sol.beta0 = np.linalg.lstsq(A,y,rcond=None)[0] # lstsq > least squares estimation with parameters beta1 and beta0
            return (0.4-sol.beta0)**2+(-0.1-sol.beta1)**2 #The function we want to minimize
        guess = [.5]*2
        bounds = [(0,1), (0,10)]
        result = optimize.minimize(objective, guess, args = (self), method = 'Nelder-Mead', bounds=bounds) #minimizer
        