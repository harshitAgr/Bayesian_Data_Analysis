
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.stats import t
from matplotlib import pyplot as plt
from scipy.stats import beta

y=np.genfromtxt('/Users/harshit/Desktop/fall 2017/Bayesian/Assignment3/windshieldy1.txt')
n=len(y)
mu=np.mean(y)
s=np.std(y)

x=np.linspace(0,30,1000)
sig=s / np.sqrt(n) #scaling for standard deviation
interval = t.interval(0.95, n, mu, sig) #95 percent confidence interval
print(interval)

samples= t.rvs(n, loc=mu, scale=sig, size=1000)
mean= np.mean(samples)
print('point estimate=',mean)

plt.figure()
plt.plot(x, t.pdf(x, n, loc=mu, scale=sig))
plt.xlim([10,20])
plt.xlabel('mu')
plt.ylabel('p(mu)')
plt.title('PDF of posterior distribution')
plt.plot((mean,mean),(0,1),color="green",label="mean")
plt.plot((interval[0],interval[0]),(0,1),color="red",label="95% confidence")
plt.plot((interval[1],interval[1]),(0,1),color="red")
plt.legend(loc=2)
plt.show()

#posterior predictive distribution
new_sig=s*np.sqrt(1+1/(n))

interval_b = t.interval(0.95, n, mu, new_sig)
print(interval_b)

samples= t.rvs(n, loc=mu, scale=new_sig, size=1000)
point_estimate_b= np.mean(samples)
print(point_estimate_b)

plt.figure()
plt.plot(x, t.pdf(x, n, loc=mu, scale=new_sig))
plt.xlim([5,25])
plt.xlabel('hardness predictions')
plt.ylabel('probabilities')
plt.title('PDF of posterior predictive distribution')
plt.plot((mean,mean),(0,1),color="green",label="mean")
plt.plot((interval[0],interval[0]),(0,1),color="red",label="95% confidence")
plt.plot((interval[1],interval[1]),(0,1),color="red")
plt.legend(loc=2)
plt.show()

