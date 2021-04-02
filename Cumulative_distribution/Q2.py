#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:38:06 2020

@author: oliverk
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from scipy.special import factorial
from scipy.integrate import quad
import sympy as sp
from sklearn import preprocessing


L0 = 1.4*(10)
a = -0.7

def lum(L):
    return ((L/L0)**a)*np.exp(-L/L0)*(1/L0)

norm = scp.integrate.quad(lum, 0, np.inf, args=())[0]

def integrand(L):
    return ((L/L0)**(a+1))*np.exp(-L/L0)*(1/norm)

def expectation():
    return scp.integrate.quad(integrand, 0, np.inf, args=())[0]

print('Expectation value:', expectation())
print('Expectation value 2:', L0*scp.special.gamma(2+a)/norm)

data= np.array([1.39, 1.40, 1.29, 5.95, 2.97, 1.63, 1.17, 2.06, 4.69, 2.48])
l = np.array(sorted(data))

def C(List, z):
    return np.sum(lum(List[:z]))/norm

y = np.arange(0, 10, 1)

c_dist = np.array([C(l, v) for v in y])

x = np.linspace(1, 6, 100)

def c_int(v):
    return scp.integrate.quad(lum, 0.0001, v, args=())[0]

t=np.linspace(0, 6, 100)

c_int_dist = ([(1/norm)*c_int(u) for u in t])

fig = plt.figure()
plt.plot(l, c_dist, 'x', color='red', label='data')
plt.plot(t, c_int_dist, '-', color='green', label='expectation')
plt.ylabel('Cumulative density')
plt.xlabel('luminosity')
plt.legend()
plt.savefig('Q2 cumulative plots')
plt.show()

print(np.max([np.linalg.norm(c_dist[o] - c_int_dist[o]) for o in y]))
                                                                                                                 #yutdfx
