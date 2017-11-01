
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
#Bayesian Data Analysis Assignment 2, exercise 1
#Read the source file
algae = open("algae.txt",mode='r')
algae_status = []
Y = 0
N = 274
for line in algae:
algae_status.append(line.rstrip())
if line.rstrip() == "1":
Y = Y+1
#Prior parameters
a = 2
b = 10
#Formulate and plot prior
x = np.linspace(0,1,1001)
plt.figure()
plt.plot(x, beta.pdf(x, a, b))
plt.xlabel('pi')
plt.ylabel('Beta(2,10)')
plt.title('Prior distribution')
#Formulate posterior
a_post = Y+a
b_post = N-Y+b
posterior = beta(a_post, b_post)
mean = beta.stats(a_post, b_post)
conf_95 = beta.interval(0.95,a_post,b_post)
#And plot
plt.figure()
plt.plot(x, posterior.pdf(x))
plt.xlim([0.05,0.25])
plt.xlabel('pi')
plt.ylabel('p(pi|y)')
plt.title('PDF of posterior distribution')
plt.plot((mean[0],mean[0]),(0,20),color="green",label="mean")
plt.plot((conf_95[0],conf_95[0]),(0,20),color="red",label="95% confidence")
plt.plot((conf_95[1],conf_95[1]),(0,20),color="red")
plt.legend(loc=2)
plt.figure()
post_cdf = posterior.cdf(x)
plt.plot(x,post_cdf)
plt.xlim([0.05,0.25])
plt.xlabel('pi')
plt.ylabel('p(pi|y)')
plt.plot((0.2,0.2),(0,1),color="red")
plt.title('CDF of posterior distribution')
#Print required information
print("Mean: %s" %mean[0])
print("95-confidence: From %s to %s" %(conf_95[0], conf_95[1]))
print("P(pi<0.2): %s" %post_cdf[200])
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
N = 274
Y = 44
#Formulate and plot priors and
#Find confidence intervals and means for posterior distributions
confs = []
means = []
x = np.linspace(0,1,1001)
plt.figure()
for a in [1,2,3,4]:
b = a*5
a_post = Y+a
b_post = N-Y+b
plt.plot(x, beta.pdf(x, a, b),label="Beta(%s,%s)" %(a,b))
confs.append(beta.interval(0.95,a_post,b_post))
means.append(beta.stats(a_post, b_post)[0])
plt.xlabel('pi')
plt.ylabel('p(pi)')
plt.title('Prior distributions')
plt.legend()
print(confs)
print(means)
plt.show()

