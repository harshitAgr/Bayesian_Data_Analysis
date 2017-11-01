
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.stats import t
from matplotlib import pyplot as plt
from scipy.stats import beta
n1=674 #P0 posterior
y1=39
a=1
b=1
a1=y1+a
b1=n1-y1+b
x=np.linspace(0,1,1000)
n2=680 #P1 posterior
y2=22
a2=y2+a
b2=n2-y2+b
p0 =np.random.beta(a1,b1,1000)
p1= np.random.beta(a2,b2,1000)
p_final = []
for i in range(0,1000):
p_final.append( (p1[i]/(1-p1[i]))/(p0[i]/(1-p0[i])) )#odd ratio
print('mean=',np.mean(p_final)) #point estimate
interval_l=np.percentile(p_final,2.5)
interval_r=np.percentile(p_final,97.5)
print('interval estimate=',(interval_l,interval_r))#interval estimate
plt.xlabel('odd ratio')
plt.ylabel('probability')
plt.hist(p_final)
plt.show()

