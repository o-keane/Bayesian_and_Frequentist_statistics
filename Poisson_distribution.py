#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:05:11 2020

@author: oliverk
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from scipy.special import factorial
from scipy.integrate import quad
import sympy as sp



s = np.linspace(0, 20, 500)
A = 10
mu = s*A               #must be a dimensionless property apparently 
Smax = 100


def d(mu, n):
    return (1/Smax)*np.exp(-mu)*(mu**n)/factorial(n)

def Z(mu, n):
    return (scp.integrate.quad(d, 0, np.inf, args = n)[0])

Zvalue1 = Z(mu, 50)
Zvalue2 = Z(mu, 5)

#fig = plt.figure()
#plt.title('Posterior poisson distribution')
#plt.plot(s, d(s, 50)/Zvalue, 'bs')
#plt.plot(s, d(s, 50)/Zvalue, '--', color='red')

fig = plt.figure()
ax1 = fig.add_subplot()
plt.title('Posterior poisson distribution')
plt.plot(s, d(mu, 50)/Zvalue1, 'bs')
plt.plot(s, d(mu, 50)/Zvalue1, '--', color='red')
ax1.set_ylabel('Posterior PDF')
ax1.set_xlabel('Density of Stars S / deg squared')

plt.savefig("Poisson distribution.pdf")
plt.show()

max_y = np.max(d(mu, 50))
#print("peak y value:", max_y)
max_x = s[d(mu, 50).argmax()]
print("Mode:", round(max_x))

x = 7.7
sig = 0.3

def d2(mu, n):
    return (1/(Smax)**2)*(np.exp(-mu)*(mu**n)/factorial(n))*(1/np.sqrt(2*np.pi*(sig**2)))*np.exp((-1/2)*(((x-(mu/10))/sig)**2))
            
def Z2(mu, n):
    return (scp.integrate.quad(d2, 0, np.inf, args = n)[0])

fig2 = plt.figure()
ax2 = fig2.add_subplot()
plt.title('Updated posterior distribution')
plt.plot(s, d2(mu, 50)*(1/Z2(mu, 50)), 'bs')
plt.plot(s, d2(mu, 50)*(1/Z2(mu, 50)), '--', color='red')
ax2.set_ylabel('Posterior PDF')
ax2.set_xlabel('Density of stars S / deg squared')

plt.savefig("Updated posterior distribution.pdf")
plt.show()
max_y2 = np.max(d2(mu, 50))
#print("peak y value:", max_y)
max_x2 = s[d2(mu, 50).argmax()]
print("Mode:", round(max_x2))
