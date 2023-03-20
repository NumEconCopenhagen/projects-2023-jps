def square(x):
    """ square numpy array
    
    Args:
    
        x (ndarray): input array
        
    Returns:
    
        y (ndarray): output array
    
    """
    
    y = x**2
    return y


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
            H = (HM+1e-10)**(1-par.alpha)*(HF+1e-10)**par.alpha
        else:
            H = ((1-par.alpha+1e-10)*(HM+1e-10)**((par.sigma-1+1e-10)/(par.sigma+1e-10))+(par.alpha+1e-10)*(HF+1e-10)**((par.sigma-1+1e-10)/(par.sigma+1e-10)))**((par.sigma+1e-10)/(par.sigma-1+1e-10))
        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

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
        guess = [5]*4
        result = optimize.minimize(obj, guess, method='SLSQP',bounds=bounds)
        opt = SimpleNamespace()
        opt.LM = result.x[0]
        opt.HM = result.x[1]
        opt.LF = result.x[2]
        opt.HF = result.x[3]
        
        return opt

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol

        for n, i in enumerate(par.wF_vec) :
            par.wF = i
            out = self.solve()
            sol.LM_vec[n] = out.LM
            sol.LF_vec[n] = out.LF
            sol.HM_vec[n] = out.HM
            sol.HF_vec[n] = out.HF

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/(sol.HM_vec+1e-10))
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """
        def objective(x, self):
            par = self.par
            sol=self.sol
            par.alpha = x[0]
            par.sigma = x[1]
            self.solve_wF_vec()
            self.run_regression()
            return (0.4-sol.beta0)**2+(-0.1-sol.beta1)**2
        guess = [.1]*2
        bounds = [(0,1), (0,10)]
        result = optimize.minimize(objective, guess, args = (self), method = 'SLSQP', bounds=bounds)
    

model = HouseholdSpecializationModelClass()



model.solve_discrete()
print(model.solve_discrete())

list_alphas = [ 0.25, 0.5, 0.75]
list_sigmas = [ 0.5, 1.0, 1.5]
results_ratio = {}


#Solving
for sigma in list_sigmas:
    model.par.sigma = sigma
    for alpha in list_alphas:
        model.par.alpha = alpha
        opt = model.solve_discrete()
        relative_hours = opt.HF/opt.HM
        results_ratio[(alpha, sigma)] = opt.HF/opt.HM

        print(f' Sigma = {model.par.sigma:.2f} Alpha = {model.par.alpha:.2f}    Ratio: {relative_hours:.2f}')



#Plotting
fig, ax = plt.subplots()
for sigma in list_sigmas:
    y = [results_ratio[(alpha, sigma)] for alpha in list_alphas]
    ax.plot(list_alphas, y, label=f"Sigma={sigma}")
ax.set_xlabel("alpha")
ax.set_ylabel("HF/HM ratio")
ax.legend()
plt.show()

results_workratio = np.empty(5)
results_wageratio = np.empty(5)

list_wF = [0.8, 0.9, 1.0, 1.1, 1.2]

for i, wF in enumerate(list_wF):
    model.par.wF = wF
    opt = model.solve_discrete()
    results_workratio[i] = np.log(opt.HF/opt.HM)
    results_wageratio[i] = np.log(model.par.wF/model.par.wM)


print(results_workratio)
print(results_wageratio)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(results_wageratio, results_workratio)
ax.set_title('Home work ratio as function of relative wage')
ax.set_xlabel('log(wF/wM)')
ax.set_ylabel('log(HF/HM)');


# code
model.par.wF = 1
model.par.sigma = 1
model.par.alpha = 0.5

# model.solve()
results_con_workratio = np.empty(5)
results_con_wageratio = np.empty(5)

for i, wF in enumerate(list_wF):
    model.par.wF = wF
    opt = model.solve()
    results_con_workratio[i] = np.log(opt.HF/opt.HM)
    results_con_wageratio[i] = np.log(model.par.wF/model.par.wM)
    print(f'LM = {opt.LM}, HM = {opt.HM}, LF = {opt.LF}, LH = {opt.HF} ')

print(results_con_workratio)
print(results_con_wageratio)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(results_con_wageratio, results_con_workratio)
ax.set_title('Home work ratio as function of relative wage')
ax.set_xlabel('log(wF/wM)')
ax.set_ylabel('log(HF/HM)');