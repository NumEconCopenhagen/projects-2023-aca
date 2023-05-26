
from types import SimpleNamespace

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class DemandClass:
    def __init__(self):
        self.rho = 0.90
        self.iota = 0.01
        self.mean_epsilon_sq = -0.5 * 0.10**2
        self.sigma_epsilon = 0.10
        self.R = (1 + 0.01)**(1/12)
        self.T = 120
        self.w = 1.0
        self.eta = 0.5
        self.q_3 = 0.
        self.Delta = 0.2
        self.simK = 5000
        self.simT = 120
        self.h_v = np.empty(self.simK)

    def kappa(self, kappa_lag, epsilon):
        return self.rho * kappa_lag + epsilon

    def policy(self, kappa, l_lag):
        l_opt = ((1 - self.eta) * kappa / self.w)**(1 / self.eta)

        if self.q_3 == 0.:
            return l_opt
        else: 
            if abs(l_lag - l_opt) > self.Delta:
                return l_opt
            else:
                return l_lag

    def profit(self, l, l_lag, kappa):
        if l == l_lag:
            iota_ = 0.
        else:
            iota_ = self.iota

        return kappa * l**(1 - self.eta) - self.w * l - iota_

    def simulate(self, do_print=False):
        np.random.seed(1917)

        for k in range(self.simK):
            epsilon_v = np.random.normal(loc=self.mean_epsilon_sq, scale=self.sigma_epsilon, size=self.simT)
            kappa_v = np.empty(self.simT)
            l_v = np.empty(self.simT)
            l_lag_v = np.empty(self.simT)
            profit_v = np.empty(self.simT)

            kappa_v[0] = np.exp(epsilon_v[0])
            l_v[0] = self.policy(kappa_v[0], 0)  # Assuming initial policy is l_{-1} = 0

            for t in range(1, self.simT):
                kappa_v[t] = self.kappa(kappa_v[t-1], np.exp(epsilon_v[t]))
                l_v[t] = self.policy(kappa_v[t], l_v[t-1])
                l_lag_v[t] = l_v[t-1]
                profit_v[t] = self.profit(l_v[t], l_lag_v[t], kappa_v[t])

            self.h_v[k] = np.sum(profit_v)

        H = np.mean(self.h_v)
        if do_print:
            print("Salon expected value with K = 5000 iterations:", round(H, 1))
    
    def find_optimal_delta(self, delta_values):
        H_values = []  # To store the H values for different delta values

        for delta in delta_values:
            self.Delta = delta
            self.simulate()
            H_values.append(np.mean(self.h_v))

        optimal_delta = delta_values[np.argmax(H_values)]
        optimal_H = H_values[np.argmax(H_values)]

        # Plot the results
        plt.plot(delta_values, H_values)
        plt.xlabel('Delta')
        plt.ylabel('H')
        plt.title('Optimal Delta for Maximizing H')
        plt.grid(True)
        plt.show()

        return optimal_delta, optimal_H
    
    

        