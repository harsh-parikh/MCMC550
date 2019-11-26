import time
import numpy as np
import pymc3 as pm
from pmcmc import pMCMC_Bench

if __name__ == "__main__":

    # True parameter values
    alpha, sigma = 1, 1
    beta = [1, 2.5]
    block_size = 2**9
    n_iter = 2**10

    for n_samples in 2**np.arange(9,14):


        # Predictor variable
        X1 = np.random.randn(n_samples)
        X2 = np.random.randn(n_samples) * 0.2

        # Simulate outcome variable
        Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(n_samples)*sigma

        #Initializing model
        basic_model = pm.Model()

        with basic_model:

            # Priors for unknown model parameters
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
            sigma = pm.HalfNormal('sigma', sigma=1)

            # Expected value of outcome
            mu = alpha + beta[0]*X1 + beta[1]*X2

            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)

            # instantiate sampler
            step = pm.Slice()

            start = time.time()
            trace = pm.sample(n_iter, step=step) 
            end = time.time()
            print(end-start)


        X = np.array([X1,X2]).T
        pmh = pMCMC_Bench(X,Y,block_size,n_iter)

        start = time.time()
        output = pmh.launch()
        end = time.time()
        print(end-start)
        print()