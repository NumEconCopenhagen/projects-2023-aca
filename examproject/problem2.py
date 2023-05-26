
from types import SimpleNamespace

import numpy as np

class demandClass:

    def __init__(self):

        par = self.par = SimpleNamespace() # parameters
        sim = self.sim = SimpleNamespace() # simulation variables

        par.rho = 0.90
        par.iota = 0.01
        par.mean_epsilon_sq = -0.5*0.10**2
        par.sigma_epsilon = 0.10
        par.R = (1 + 0.01)**(1/12)
        par.T = 120
        par.w = 1.0
        par.eta = 0.5
        
        par.q_3 = 0.
        par.pol = 0.2

        # Set number of iterations for simulation
        par.simK = 5000

        # Set number of periods for simulation
        par.simT = 120

        # Set number of interations for shock series
        sim.h_values = np.empty(par.simK)

         # Simulation vector
        sim.epsilon_v = np.empty(par.simT)

        sim.kappa_v = np.empty(par.simT)
        sim.l_v = np.empty(par.simT)
        sim.l_lag_v = np.empty(par.simT)

        sim.profit_v = np.empty(par.simT)

     # Define kappa (demand shock) function
    def kappa(self,kappa_lag,epsilon):
        par = self.par
        return par.rho*(kappa_lag) + epsilon

    # Define policy function
    def policy(self,kappa):
        par = self.par
        return ((1-par.eta)*kappa/par.w)**(1/par.eta)

    # Define profit function
    def profit(self,l,l_lag,kappa):
        par = self.par
        if l == l_lag:
            iota_ = 0.
        else: 
            iota_ = par.iota

        return kappa*l**(1-par.eta) - par.w*l - iota_
    
    def simulate(self):
        np.random.seed(1917)

        par = self.par
        sim = self.sim

        # Simulate epsilon shocks with K iterations across T periods
        for k in range(par.simK):
            sim.epsilon_v = np.random.normal(loc=par.mean_epsilon_sq, scale=par.sigma_epsilon, size=par.simT)
            
            # Set initial values
            sim.kappa_v[0] = 1.0 + np.exp(sim.epsilon_v[0])
        
             # Find values for t>0
            if par.q_3 == 0.: #question 1-2 with only the orginal policy function
                for t in range(1,par.simT):
                    sim.kappa_v[t] = self.kappa(sim.kappa_v[t-1], np.exp(sim.epsilon_v[t]))
                    sim.l_v[t] = self.policy(sim.kappa_v[t])
                    sim.l_lag_v[t] = np.concatenate(([0], sim.l_v[:-1]))[t] # 0 put in as first value 
                    
                    sim.profit_v[t] = par.R**(-t)*self.profit(sim.l_v[t], sim.l_lag_v[t], sim.kappa_v[t])

            else: #question 3-5 with the new policy function 
                for t in range(1,par.simT):
                    sim.kappa_v[t] = self.kappa(sim.kappa_v[t-1], np.exp(sim.epsilon_v[t]))

                    if np.abs(sim.l_v[t-1]-self.policy(sim.kappa_v[t])) > par.pol:
                        sim.l_v[t] = self.policy(sim.kappa_v[t]) # use policy function
                        sim.l_lag_v[t] = np.concatenate(([0], sim.l_v[:-1]))[t] # 0 put in as first value 
                    else: 
                        sim.l_v[t] = sim.l_v[t-1] # use lagged employment 
                        sim.l_lag_v[t] = np.concatenate(([0], sim.l_v[:-1]))[t] # 0 put in as first value 
                
                    sim.profit_v[t] = par.R**(-t)*self.profit(sim.l_v[t], sim.l_lag_v[t], sim.kappa_v[t])

            # Save profits for each iteration 
            sim.h_values[k] = np.sum(sim.profit_v)

        # Take mean of all iterations and find expected profits
        H = np.mean(sim.h_values)

        print("Salon expected value with K = 5000 iterations:", round(H, 1))
