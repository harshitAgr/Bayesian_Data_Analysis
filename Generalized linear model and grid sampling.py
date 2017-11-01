
# coding: utf-8

# In[ ]:

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from scipy.special import expit  # aka logistic
from mpl_toolkits.mplot3d import Axes3D


# seed a random state
rng = np.random.RandomState(0)

x = (-.863, -.296, -.053, .7270)
n = (5, 5, 5, 5)
y = (0, 1, 3, 5)

def logitinv( x ):
    return 1/(1+np.exp(-x))

def prior( a,b ):
    return np.exp(-1/1.5*((a*a)/4+((b-10)*(b-10))/100-a*(b-10)/20))

def likelihood( a,b ):
    L=1
    for i in range(1,len(x)):
        L = L*logitinv(a+b*x[i])**y[i]*(1.0-logitinv(a+b*x[i]))**(n[i]-y[i])
    return L

def pd( a,b ):
    return prior(a,b)*likelihood(a,b);

#a1 - posterior at grid points

alpha = np.linspace(-5.0000,10.0000,100)
beta = np.linspace(-5.0000,30.0000,100)

na = len(alpha)
nb = len(beta)
z = np.ndarray((na, nb))
for i in range(1,na):
    z[i] = pd(alpha, beta[i])

#a2 - 1000 samples  
nsamp = 1000
samp_indices = np.unravel_index(
    rng.choice(z.size, size=nsamp, p=z.ravel()/np.sum(z)),
    z.shape
)
alpha_sample = alpha[samp_indices[1]]
beta_sample = beta[samp_indices[0]]
    
    
    # add random jitter, see BDA3 p. 76
alpha_sample += (rng.rand(1000) - 0.5) * (alpha[1]-alpha[0])
beta_sample += (rng.rand(1000) - 0.5) * (beta[1]-beta[0])
  
#a3 - posterior contour plot for alpha and beta

plt.contour(alpha,beta,z,21)
plt.xlabel("alpha")
plt.ylabel("beta")
plt.title("The contour plot of the posterior distribution")
plt.show()

#a4 - scatter plot of the 1000 samples


plt.scatter(alpha_sample, beta_sample, 10, linewidth=0)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.title('Scatter plot of samples')
plt.xlim([-5,10])
plt.ylim([-5,30])
plt.show()

#a5 - LD50 distribution

bpi = beta_sample > 0
samp_ld50 = -alpha_sample[bpi]/beta_sample[bpi]

plt.subplot(3,1,3)
plt.hist(samp_ld50, np.arange(-0.5, 0.5, 0.02))
plt.xlim([-0.5, 0.5])
plt.xlabel(r'LD50 = -$\alpha/\beta$')
plt.yticks(())
plt.title('histogram of LD50')
plt.show()

#part b1 - prior distribution

z1 = np.ndarray((na, nb))
for i in range(1,na):
    z1[i] = prior(alpha, beta[i])
plt.contour(alpha,beta,z1,21)
plt.xlabel("alpha")
plt.ylabel("beta")
plt.title("The contour plot of the prior distribution")
plt.show()

#part b2 - likelihood distribution

z2 = np.ndarray((na, nb))
for i in range(1,na):
    z2[i] = likelihood(alpha, beta[i])
plt.contour(alpha,beta,z2,21)
plt.xlabel("alpha")
plt.ylabel("beta")
plt.title("The contour plot of the likelihood distribution")
plt.show()

