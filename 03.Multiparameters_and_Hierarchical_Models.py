#######################################################################################
###                 03.Multiparameters_and_Hierarchical_Models.py                   ###
#######################################################################################

import numpy as np
import pymc3 as pm
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

# >>>>>>>>>>>>>>>>>>  joint and marginal distribution  <<<<<<<<<<<<<<<<<<<<< #
np.random.seed(123)
x = np.random.gamma(2, 1, 1000)
y = np.random.normal(0, 1, 1000)
data = pd.DataFrame(data=np.array([x, y]).T, columns=['$\\theta_1$', '$\\theta_2$'])
sns.jointplot(x='$\\theta_1$', y='$\\theta_2$', data=data, stat_func=None)
plt.show()

# >>>>>>>>>>>>>>>>>>  Gaussian inferences  <<<<<<<<<<<<<<<<<<<<< #
data = np.array([51.06, 55.12, 53.73, 50.24, 52.05, 56.40, 48.45, 52.34, 55.65, 51.49, 51.86, 63.43, 53.00, 56.09, 51.93, 52.31, 52.33, 57.48, 57.44, 55.14, 53.93, 54.62, 56.09, 68.58, 51.36, 55.47, 50.73, 51.94, 54.95, 50.39, 52.91, 51.5, 52.68, 47.72, 49.73, 51.82, 54.99, 52.84, 53.19, 54.52, 51.46, 53.73, 51.61, 49.81, 52.42, 54.3, 53.84, 53.16])
quantile = np.percentile(data, [25, 75])
iqr = quantile[1] - quantile[0]
upper = quantile[1] + iqr * 1.5
lower = quantile[0] - iqr * 1.5
clean_data = data[(data > lower) & (data < upper)]
sns.kdeplot(data)
plt.xlabel('$x$', fontsize=16)

with pm.Model() as Gaussian_model:
    mu = pm.Uniform('mu', lower=40, upper=70)
    sigma = pm.HalfNormal('sigma', sd=10)
    y = pm.Normal('y', mu=mu, sd=sigma, observed=data)
    trace = pm.sample(1100)
chain = trace[100:]
pm.traceplot(chain)

# summarize
pm.summary(chain)

# predictive sample
y_pred = pm.sample_ppc(chain, 100, Gaussian_model, size=len(data))
sns.kdeplot(data, color='b')
for draw in y_pred['y']:
    sns.kdeplot(draw.flatten(), color='r', alpha=0.1)
plt.title('Gaussian model', fontsize=16)
plt.xlabel('$x$', fontsize=16)

>>>>>>>>>>>>>>>>>>  Gaussian robust inferences  <<<<<<<<<<<<<<<<<<<<< #
plt.figure(figsize=(8, 6))
x = np.linspace(-10, 10, 200)
for df in [1, 2, 5, 30]:
    dist = stats.t(df)
    y = dist.pdf(x)
    plt.plot(x, y, label=r'$\nu$ = {}'.format(df))
y = stats.norm.pdf(x)
plt.plot(x, y, label=r'$\nu = \infty$')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$pdf(x)$', fontsize=16, rotation=90)
plt.legend(loc=0, fontsize=14)
plt.xlim(-7, 7)

with pm.Model() as t_model:
    mu = pm.Uniform('mu', 40, 75)
    sigma = pm.HalfNormal('sigma', sd=10)
    nu = pm.Exponential('nu', 1/30)
    y = pm.StudentT('y', mu=mu, sd=sigma, nu=nu, observed=data)
    trace = pm.sample(1100)
chain = trace[100:]
# pm.traceplot(chain)

# summarize
pm.summary(chain)

# predictive sample
y_pred = pm.sample_ppc(chain, 100, t_model, size=len(data))
sns.kdeplot(data, color='b')
for draw in y_pred['y']:
    sns.kdeplot(draw.flatten(), color='r', alpha=0.1)
plt.title('Student\'s t model', fontsize=16)
plt.xlabel('$x$', fontsize=16)
plt.xlim(35, 75)

# >>>>>>>>>>>>>>>>>>  Comparing groups  <<<<<<<<<<<<<<<<<<<<< #
tips = sns.load_dataset('tips')
tips.tail()

sns.violinplot(x='day', y='tip', data=tips)

y = tips['tip'].values
x = pd.Categorical(tips['day']).codes

with pm.Model() as groups_model:
    mus = pm.Normal('mus', mu=0, sd=10, shape=len(set(x)))
    sds = pm.HalfNormal('sds', sd=10, shape=len(set(x)))
    y = pm.Normal('y', mu=mus[x], sd=sds[x], observed=y)
    trace = pm.sample(5000)

chain = trace[100::]
pm.traceplot(chain)

# summarize
pm.summary(chain)

# infer difference
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
comparisons = [(i,j) for i in range(4) for j in range(i+1, 4)]
pos = [(k,l) for k in range(3) for l in (0, 1)]

for (i,j), (k,l) in zip(comparisons, pos):
    mus_diff = chain['mus'][:,i] - chain['mus'][:,j]
    d_cohen = (mus_diff / np.sqrt((chain['sds'][:,i]**2 + chain['sds'][:,j]**2) / 2)).mean()

    ps = stats.norm.cdf(d_cohen/(2**0.5))
    pm.plot_posterior(mus_diff, ref_val=0, ax=axes[k,l])
    axes[k,l].plot(0, label="Cohen's d = {:.2f}\nProb sup = {:.2f}".format(d_cohen, ps) ,alpha=0)
    axes[k,l].set_xlabel('$\mu_{}-\mu_{}$'.format(i, j), fontsize=18)
    axes[k,l].legend(loc=0, fontsize=14)
plt.tight_layout()

# >>>>>>>>>>>>>>>>>>  Hierarchical Models  <<<<<<<<<<<<<<<<<<<<< #
N_samples =  [30, 30, 30]
G_samples =  [18, 18, 18]

group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []  
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))

with pm.Model() as h_model:
    alpha = pm.HalfCauchy('alpha', beta=10)
    beta = pm.HalfCauchy('beta', beta=10)
    theta = pm.Beta('theta', alpha=alpha, beta=beta, shape=len(N_samples))
    y = pm.Bernoulli('y', p=theta[group_idx], observed=data)
    trace = pm.sample(2000)
chain = trace[200:]
pm.traceplot(chain)

pm.summary(chain)

# shrinkage
x = np.linspace(0, 1, 100)
for i in np.random.randint(0, len(chain), size=100):
    pdf = stats.beta(chain['alpha'][i], chain['beta'][i]).pdf(x)
    plt.plot(x, pdf,  'g', alpha=0.1)

dist = stats.beta(chain['alpha'].mean(), chain['beta'].mean())
y = dist.pdf(x)
mode = x[np.argmax(pdf)]
mean = dist.moment(1)
plt.plot(x, y, label='mode = {:.2f}\nmean = {:.2f}'.format(mode, mean))

plt.legend(fontsize=14)
plt.xlabel('$\\theta_{prior}$', fontsize=16)
plt.tight_layout()