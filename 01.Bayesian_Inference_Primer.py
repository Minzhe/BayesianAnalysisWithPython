#########################################################################
###                 01.Bayesian_Inference_Primer.py                   ###
#########################################################################

import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')


# >>>>>>>>>>>>>>>>>>  normal distribution  <<<<<<<<<<<<<<<<<<<<< #
mus = [-1, 0, 1]
sds = [0.5, 1, 1.5]
x = np.linspace(-7, 7, 100)
fig, axes = plt.subplots(len(mus), len(sds), sharex=True, sharey=True)
for i in range(len(mus)): 
    for j in range(len(sds)):
        mu, sd = mus[i], sds[j]
        y = stats.norm(mu, sd).pdf(x)
        axes[i,j].plot(x, y)
        axes[i,j].plot(0, 0, label='$\\mu$ = {:3.2f}\n$\\sigma$ = {:3.2f}'.format(mu, sd), alpha=0)
        axes[i,j].legend(fontsize=12)
axes[2,1].set_xlabel('$x$', fontsize=16)
axes[1,0].set_ylabel('$pdf(x)$', fontsize=16)
plt.tight_layout()
plt.show()

# >>>>>>>>>>>>>>>>>>  series data (non-iid)  <<<<<<<<<<<<<<<<<<<<< #
data = np.genfromtxt('data/01.mauna_loa_CO2.csv', delimiter=',')
plt.plot(data[:,0], data[:,1])
plt.xlabel('$year$', fontsize=16)
plt.ylabel('CO_2 (ppmv)$', fontsize=16)
plt.show()

>>>>>>>>>>>>>>>>>>  binorminal distribution  <<<<<<<<<<<<<<<<<<<<< #
ns = [1, 2, 4]
ps = [0.25, 0.5, 0.75]
x = np.arange(0, max(ns)+1)
fig, axes = plt.subplots(len(ns), len(ps), sharex=True, sharey=True)
for i in range(len(ns)):
    for j in range(len(ps)):
        n, p = ns[i], ps[j]
        y = stats.binom(n, p).pmf(x)
        axes[i,j].vlines(x, 0, y, colors='b', lw=5)
        axes[i,j].set_ylim(0, 1)
        axes[i,j].plot(0, 0, label='n = {:3.2f}\np = {:3.2f}'.format(n, p), alpha=0)
        axes[i,j].legend(fontsize=12)
axes[2,1].set_xlabel('$\\theta$', fontsize=14)
axes[1,0].set_ylabel('$p(y|\\theta)$', fontsize=14)
plt.show()

# >>>>>>>>>>>>>>>>>>  beta distribution  <<<<<<<<<<<<<<<<<<<<< #
params = [0.5, 1, 2, 3]
x = np.linspace(0, 1, 100)
fig, axes = plt.subplots(len(params), len(params), sharex=True, sharey=True)
for i in range(len(params)):
    for j in range(len(params)):
        a, b = params[i], params[j]
        y = stats.beta(a, b).pdf(x)
        axes[i,j].plot(x, y)
        axes[i,j].plot(0, 0, label="$\\alpha$ = {:3.2f}\n$\\beta$ = {:3.2f}".format(a, b), alpha=0)
        axes[i,j].legend(fontsize=12)
axes[3,0].set_xlabel('$\\theta$', fontsize=14)
axes[0,0].set_ylabel('$p(\\theta)$', fontsize=14)
plt.show()

# >>>>>>>>>>>>>>>>>>  posterior distribution  <<<<<<<<<<<<<<<<<<<<< #
theta_true = 0.35
trials = [0, 1, 2, 3, 4, 8, 16, 32, 50, 150]
heads = [0, 1, 1, 1, 1, 4, 6, 9, 13, 48]

priors = [(1, 1), (0.5, 0.5), (20, 20)]
x = np.linspace(0, 1, 100)

for idx, N in enumerate(trials):
    if idx == 0:
        plt.subplot(4, 3, 2)
    else:
        plt.subplot(4, 3, idx+3)
    y = heads[idx]
    for (a_prior, b_prior), c in zip(priors, ('b', 'r', 'g')):
        p_post = stats.beta(a_prior + y, b_prior + N - y).pdf(x)
        plt.plot(x, p_post, color=c)
        plt.fill_between(x, 0, p_post, color=c, alpha=0.6)
    
    plt.axvline(theta_true, ymax=0.3, color='k')
    plt.plot(0, 0, label='{:d} experiments\n{:d} heads'.format(N, y), alpha=0)
    plt.xlim(0,1)
    plt.ylim(0,12)
    plt.xlabel(r"$\theta$") 
    plt.legend()
    plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()

# >>>>>>>>>>>>>>>>>>  highest posterior density (HPD)  <<<<<<<<<<<<<<<<<<<<< #
def naiveHPD(post):
    sns.kdeplot(post)
    HPD = np.percentile(post, [2.5, 97.5])
    plt.plot(HPD, [0, 0], label='HPD {:.2f} {:.2f}'.format(*HPD), linewidth=8, color='k')
    plt.legend(fontsize=16)
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.gca().axes.get_yaxis().set_ticks([])

np.random.seed(1)
post = stats.beta.rvs(5, 11, size=1000)
naiveHPD(post)
plt.xlim(0,1)
plt.show()

np.random.seed(1)
gauss_a = stats.norm.rvs(loc=4, scale=0.9, size=3000)
gauss_b = stats.norm.rvs(loc=-2, scale=1, size=2000)
mix_norm = np.concatenate([gauss_a, gauss_b])
naiveHPD(mix_norm)
plt.show()