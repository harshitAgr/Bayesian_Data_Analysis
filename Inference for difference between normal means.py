
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.stats import t
from matplotlib import pyplot as plt
from scipy.stats import beta



y1=np.genfromtxt('/Users/harshit/Desktop/fall 2017/Bayesian/Assignment3/windshieldy1.txt')
y2=np.genfromtxt('/Users/harshit/Desktop/fall 2017/Bayesian/Assignment3/windshieldy2.txt')
n1=len(y1)
print(n1)
mu1=np.mean(y1)
s1=np.std(y1)
n2=len(y2)
print(n2)
mu2=np.mean(y2)
s2=np.std(y2)
sig1=s1 / np.sqrt(n1) #scaling for standard deviation
samples1= t.rvs(n1-1, loc=mu1, scale=sig1, size=1000)
sig2=s2 / np.sqrt(n2) #scaling for standard deviation
samples2= t.rvs(n2-1, loc=mu2, scale=sig2, size=1000)
mu=samples1-samples2
interval_l=np.percentile(mu,2.5)
interval_r=np.percentile(mu,97.5)
print('interval estimate=',(interval_l,interval_r))#interval estimate
print(np.mean(mu))
plt.figure()
plt.hist(mu)
plt.xlabel('ud=u1-u2')
plt.ylabel('probabilities')
plt.show(

