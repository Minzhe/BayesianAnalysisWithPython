#########################################################################
###                        02.PyMC3.primer.py                         ###
#########################################################################

import pymc3 as pm
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>  grid computing  <<<<<<<<<<<<<<<<<<<<< #
    def posterior_grid(grid_points=100, head=6, tosses=9):
        '''
        A grid implementation for the coin-flip problem
        '''
        grid = np.linspace(0, 1, grid_points)
        # uniform prior
        prior = np.repeat(5, grid_points)
        # likelihood
        likelihood = stats.binom.pmf(head, tosses, grid)
        # posterior
        posterior = likelihood * prior
        posterior = posterior / posterior.sum()

        return grid, posterior

    points = 15
    h, n = 1, 4
    grid, posterior = posterior_grid(points, h, n)
    plt.plot(grid, posterior, 'o-')
    plt.plot(0, 0, label='heads = {}\ntosses = {}'.format(h, n), alpha=0)
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.legend(loc=0, fontsize=14)
    plt.show()

    >>>>>>>>>>>>>>>>>>  Monte Carlo  <<<<<<<<<<<<<<<<<<<<< #
    N = 10000
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2)  <= 1
    outside = np.invert(inside)
    pi = inside.sum()*4/N
    error = abs((pi - np.pi)/pi) * 100
    plt.plot(x[inside], y[inside], 'b.')
    plt.plot(x[outside], y[outside], 'r.')
    plt.plot(0, 0, label='$\hat \pi$ = {:4.3f}\nerror = {:4.3f}%'.format(pi, error), alpha=0)
    plt.axis('square')
    plt.legend(frameon=True, framealpha=0.9, fontsize=16)
    plt.show()

    # >>>>>>>>>>>>>>>>>>  simple metropolis  <<<<<<<<<<<<<<<<<<<<< #
    def metropolis(dist, steps=10000):
        '''
        A very simple Metropolis implementation
        '''
        samples = np.zeros(steps)
        x_old = dist.mean()
        y_old = dist.pdf(x_old)

        for i in range(steps):
            x_new = x_old + np.random.normal(0, 0.5)
            y_new = dist.pdf(x_new)
            acceptance = y_new / y_old
            if acceptance >= np.random.random():
                samples[i] = x_new
                x_old = x_new
                y_old = y_new
            else:
                samples[i] = x_old
        
        return samples

    np.random.seed(345)
    dist = stats.beta(0.4, 2)
    samples = metropolis(dist=dist)
    x = np.linspace(0.01, 0.99, 1000)
    y = dist.pdf(x)
    plt.xlim(0, 1)
    plt.plot(x, y, 'r-', lw=3, label='True distribution')
    plt.hist(samples, bins=100, normed=True, label='Estimated distribution')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$pdf(x)$', fontsize=14)
    plt.legend(fontsize=14)
    plt.show()

    # >>>>>>>>>>>>>>>>>>  PyMC3  <<<<<<<<<<<<<<<<<<<<< #
    np.random.seed(123)
    n = 50
    theta_true = 0.35
    data = stats.bernoulli.rvs(p=theta_true, size=n)
    print(data)

    with pm.Model() as beta_binomial:
        # prior
        theta = pm.Beta('theta', alpha=1, beta=1)
        # likelihood
        y = pm.Bernoulli('y', p=theta, observed=data)
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(1000, step=step, start=start)

    burnin = 0 # no burnin
    chain = trace[burnin:]
    pm.traceplot(chain, lines={'theta': theta_true})

    with beta_binomial:
        step = pm.Metropolis()
        multi_trace = pm.sample(1000, step=step, njobs=4)
    
    burnin = 0  # no burnin
    multi_chain = multi_trace[burnin:]
    pm.traceplot(multi_chain, lines={'theta': theta_true})

    # convergence
    pm.gelman_rubin(multi_chain)
    pm.forestplot(multi_chain, varnames=['theta'])

    # summary
    pm.summary(multi_chain)

    # autocorrelation
    pm.autocorrplot(chain)

    # effective size
    pm.effective_n(multi_chain)['theta']
    
    # Summerize the posterior
    pm.plot_posterior(chain, kde_plot=True)
    plt.show()