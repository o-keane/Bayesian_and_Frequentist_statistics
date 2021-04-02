#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 23:15:16 2020

@author: oliverk
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from scipy.special import factorial
from scipy.integrate import quad
import sympy as sp


data_10hr = np.loadtxt("data_10hr.txt")
data_24hr = np.loadtxt("data_24hr.txt")
data_100hr = np.loadtxt("data_100hr.txt")

B = 5
L0max = 10
Tmax = 100
A0max = 10


def Z0(data):
    return np.product((1/np.sqrt(2*np.pi))*np.exp((-1/2)*((data-B)**2)))

Z0_10hr = Z0(data_10hr)
Z0_24hr = Z0(data_24hr)
Z0_100hr = Z0(data_100hr)


def Z_1_integrand(L0, data):
    return (1/L0max)*np.product((1/np.sqrt(2*np.pi))*np.exp(-1/2*((data-B-L0)**2)))

def Z1(data):
    return scp.integrate.quad(Z_1_integrand, 0, L0max, args = data)[0]
    #args essentially tells us the variables not being integrated
    
Z1_10hr = Z1(data_10hr)
Z1_24hr = Z1(data_24hr)
Z1_100hr = Z1(data_100hr)

def Z2(data):
    f = lambda A0, T : (1/(A0max*Tmax))*np.product((1/np.sqrt(2*np.pi))*np.exp((-1/2)*((data-B-(A0*np.exp(-np.arange(len(data))/T)))**2)))
    i = scp.integrate.dblquad(f, 1, 100, 0, 10)     #limits of T then A0 
    return(i)[0]
    
Z2_10hr = Z2(data_10hr)
Z2_24hr = Z2(data_24hr)
Z2_100hr = Z2(data_100hr)

#Calculating Ratios

O_2_1_10hr = Z2_10hr/Z1_10hr
O_2_1_24hr = Z2_24hr/Z1_24hr
O_2_1_100hr = Z2_100hr/Z1_100hr

O_1_0_10hr = Z1_10hr/Z0_10hr
O_1_0_24hr = Z1_24hr/Z0_24hr
O_1_0_100hr = Z1_100hr/Z0_100hr

O_2_0_10hr = Z2_10hr/Z0_10hr
O_2_0_24hr = Z2_24hr/Z0_24hr
O_2_0_100hr = Z2_100hr/Z0_100hr

print("posterior odds ratio O(2,1):", "(10hr)", O_2_1_10hr, "(24hr)", O_2_1_24hr, "(100hr)", O_2_1_100hr)
print("posterior odds ratio O(1,0):", "(10hr)", O_1_0_10hr, "(24hr)", O_1_0_24hr, "(100hr)", O_1_0_100hr)
print("posterior odds ratio O(2,0):", "(10hr)", O_2_0_10hr, "(24hr)", O_2_0_24hr, "(100hr)", O_2_0_100hr)

#print("Z0:", Z0_10hr, Z0_24hr, Z0_100hr)
#print("Z1:", Z1_10hr, Z1_24hr, Z1_100hr)
#print("Z2:", Z2_10hr, Z2_24hr, Z2_100hr)

# MODEL 1 PLOT

L0list = np.linspace(0, 3, 200)

def posterior_1(L0, data, Z):
    return (1/3*Z)*(1/(A0max*Tmax))*np.product((1/np.sqrt(2*np.pi))*np.exp((-1/2)*(data-B-L0)**2))

Model1_plot_10hr = np.zeros(200)
Model1_plot_24hr = np.zeros(200)
Model1_plot_100hr = np.zeros(200)

for q in range(len(L0list)):
    L0 = L0list[q]
    Model1_plot_10hr[q] = posterior_1(L0, data_10hr, Z1_10hr)
    Model1_plot_24hr[q] = posterior_1(L0, data_24hr, Z1_24hr)
    Model1_plot_100hr[q] = posterior_1(L0, data_100hr, Z1_100hr)

fig = plt.figure(figsize=(8,8))
plt.title('Model 1 posterior pdf at 10hrs')
plt.xlabel('Parameter L0', fontsize='large')
plt.ylabel('Posterior probability', fontsize='large')
plt.plot(L0list, Model1_plot_10hr)
plt.savefig("Model 1 posterior pdf at 10hrs.pdf")


fig = plt.figure(figsize=(8,8))
plt.title('Model 1 posterior pdf at 24hrs')
plt.xlabel('Parameter L0', fontsize='large')
plt.ylabel('Posterior probablility', fontsize='large')

plt.plot(L0list, Model1_plot_24hr)
plt.savefig("Model 1 posterior pdf at 24hrs.pdf")


fig = plt.figure(figsize=(8,8))
plt.title('Model 1 posterior pdf at 100hrs')
plt.xlabel('Parameter L0', fontsize='large')
plt.ylabel('Posterior probablility', fontsize='large')

plt.plot(L0list, Model1_plot_100hr)
plt.savefig("Model 1 posterior pdf at 100hrs.pdf")
plt.show()

# MODEL 2 PLOT

A0list = np.linspace(0, 2.5, 200)
Tlist = np.linspace(0, 100, 200)

def posterior_2(A0, T, data, Z):
    return (1/3*Z)*(1/(A0max*Tmax))*np.product((1/np.sqrt(2*np.pi))*np.exp((-1/2)*((data-B-(A0*np.exp(-np.arange(len(data))/T)))**2)))

Model2_plot_10hr = np.zeros((200, 200))
Model2_plot_24hr = np.zeros((200, 200))
Model2_plot_100hr = np.zeros((200, 200))

for i in range(len(A0list)):
    A0 = A0list[i]
    for j in range(len(Tlist)):
        T = Tlist[j]
        Model2_plot_10hr[i][j] = posterior_2(A0, T, data_10hr, Z2_10hr)
        Model2_plot_24hr[i][j] = posterior_2(A0, T, data_24hr, Z2_24hr)
        Model2_plot_100hr[i][j] = posterior_2(A0, T, data_100hr, Z2_100hr)

fig = plt.figure(figsize=(8,8))
plt.title('Model 2 posterior pdf at 10hrs')
plt.xlabel('Parameter A0', fontsize='large')
plt.ylabel('Parameter T', fontsize='large')
plt.contour(A0list, Tlist, Model2_plot_10hr)
plt.savefig("Model 2 posterior pdf at 10hrs.pdf")

fig = plt.figure(figsize=(8,8))
plt.title('Model 2 posterior pdf at 24hrs')
plt.xlabel('Parameter A0', fontsize='large')
plt.ylabel('Parameter T', fontsize='large')
plt.contour(A0list, Tlist, Model2_plot_24hr)
plt.savefig("Model 2 posterior pdf at 24hrs.pdf")

fig = plt.figure(figsize=(8,8))
plt.title('Model 2 posterior pdf at 100hrs')
plt.xlabel('Parameter A0', fontsize='large')
plt.ylabel('Parameter T', fontsize='large')
plt.contour(A0list, Tlist, Model2_plot_100hr)
plt.savefig("Model 2 posterior pdf at 100hrs.pdf")

plt.show()

# DATA PLOT

L=0.45
A=0.8
t=51

time = np.linspace(1, 100, 100)

Model0 = B
Model1 = B+L
Model2 = B+A*np.exp(-np.arange(len(data_100hr))/t)

fig = plt.figure(figsize=(10,10))
plt.scatter(time, data_100hr, marker='x')
plt.hlines(Model0, 1, 100, colors='r', linestyles='solid', label='Model 0')
plt.hlines(Model1, 1, 100, colors='b', linestyles='solid', label='Model 1')
plt.plot(time, Model2, color='black', linestyle='solid', label='Model 2')
plt.xlabel('Time', fontsize='large')
plt.ylabel('Data', fontsize='large')
plt.legend(fontsize='x-large')
plt.savefig("Data plot with model predictions at 100hrs.pdf")
plt.show()
#print(f)

