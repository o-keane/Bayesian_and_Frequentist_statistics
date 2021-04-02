#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 18:38:57 2020

@author: oliverk
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from scipy.special import factorial
from scipy.integrate import quad
import sympy as sp


data = np.genfromtxt("Tdisp.dat", skip_header=1)

t = data[:,0]                       #temperature data
d = data[:,1]                       #displacement data

def mean(z):
    return np.sum(z)/len(z)

def sigma(z):
    return np.sqrt(np.sum((z-mean(z))**2)/len(z))

def pearson():                      #Pearson coefficient
    return (1/len(data))*np.sum((t-mean(t))*(d-mean(t)))/(sigma(t)*sigma(d))

print('Pearson:', pearson())

rank_t = sorted(t)
index_t = np.array([rank_t.index(v) for v in t])

rank_d = sorted(d)
index_d = np.array([rank_d.index(v) for v in d])

dif_square = (index_t - index_d)**2

def spearman():
    return 1-(6*np.sum(dif_square)/(len(data)*((len(data)**2)-1)))

print('Spearman:', spearman())

tp = (pearson())*np.sqrt(8)/np.sqrt(1-((pearson())**2))
ts = (spearman())*np.sqrt(8)/np.sqrt(1-((spearman())**2))
print('tp', tp)
print('ts', ts)

#   Plot Data

fig = plt.figure()
plt.plot(t, d, 'x', color='red')
plt.title('Data Plot')
plt.xlabel('Displacement (microns)')
plt.ylabel('Temperature (deg C)')
plt.savefig('Q1 data plot')
plt.show()

print('Prefer spearman because pearson assumes gaussian distribution which the data doesnt fit')

def mean2(x, y):
    return np.sum(x*y)/len(x)

#   Least Squares calculation and plots for measurement d against variable t 

m1 = ((mean(t)*mean(d))-mean2(t,d))/-(sigma(t)**2)
c1 =mean(d) - m1*mean(t)

R_square1= np.sum((d-m1*t-c1)**2)
print('slope 1:', m1)
print('intercept 1:', c1)
print('Linear regression 1:', R_square1)

temp = np.linspace(10, 32, 50)

fig2 = plt.figure()
plt.plot(t, d, 'x', color='red', label='data')
plt.plot(temp, (m1*temp)+c1, '--', color='blue', label='best fit')
plt.title("least squares fit 1")
plt.xlabel('Temperature (deg C)')
plt.ylabel('Displacement (microns)')
plt.legend()
plt.savefig('Q1 least squares fit 1')
plt.show()

#   Least Squares calculation and plots for measurement t against variable d

m2 = ((mean(t)*mean(d))-mean2(t,d))/-(sigma(d)**2)
c2 = mean(t) - m2*mean(d)

R_square2 = np.sum((t-m2*d-c2)**2)
print('slope 2:', m2)
print('intercept 2:', c2)
print('Linear regression 2:', R_square2)

disp = np.linspace(2.65, 3.2, 50)

fig2 = plt.figure()
plt.plot(d, t, 'x', color='red', label='data')
plt.plot(disp, (m2*disp)+c2, '--', color='blue', label='best fit')
plt.title("least squares fit 2")
plt.xlabel('Displacement (microns)')
plt.ylabel('Temperature (deg C)')
plt.legend()
plt.savefig('Q1 least squares fit 2')
plt.show()

print(1/m2)



