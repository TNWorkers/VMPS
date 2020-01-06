#!/usr/bin/env python                                                                                                                                                                          
# -*- coding: utf-8 -*-                                                                                                                                                                        
                                                                                                                                                                                               
from __future__ import print_function #python2 is then compatible with python3 print syntax. The other way around is not possible.                                                             
                                                                                                                                                                                               
import os, sys                                                                                                                                                                                 
import numpy as np                                                                                                                                                                             
#from termcolor import colored, cprint                                                                                                                                                         
import matplotlib.pyplot as plt                                                                                                                                                                
from math import *                                                                                                                                                                             
from cmath import *                                                                                                                                                                            
from matplotlib import rc, rcParams, colors                                                                                                                                                    
import h5py
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import getpass                                                                                                                                                                                 
from matplotlib import rc, rcParams                                                                                                                                                            
from scipy.optimize import curve_fit                                                                                                                                                                                               
rc('text', usetex=True)
rc('font', **{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('font', size=14)
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
rcParams.update(params)

def myround(x):
        # if x == int(x):
        #         return int(x)
        # else:
        #         return x
        if np.round(x,10)==int(x):
                return int(x)
        else:
                return np.round(x,10)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-J', action='store', default=0.0)
parser.add_argument('-L', action='store', default=20)
parser.add_argument('-Ly', action='store', default=1)
parser.add_argument('-U', action='store', default=2)
parser.add_argument('-save', action='store', default=False)
parser.add_argument('-set', action='store', default='007')
parser.add_argument('-BC', action='store', default='IBC')
args = parser.parse_args()


def second(x):
        return 0.5*np.power((-5.731-x),0.5)

# second = np.vectorize(myfunc)
# def second(x, beta, c):
#         print('x',x)
#         if x > -5.732:
#                 return 0.
#         else:
#                 return c*np.power((-5.732-x),beta)
        
L=int(args.L)
        #Vs = np.arange(-5.9,-5.69,0.01,dtype=float)
Vs = np.array([-5.1,-5.2,-5.3,-5.4,-5.5,-5.6,-5.7,-5.8,-5.9,-6])
Ts = np.zeros(len(Vs))
for iV in range(len(Vs)):
        V = Vs[iV]
        filename = ''
        if abs(V - (-6.0)) > 1.e-4:
                filename = 'obs/L=2_Ly=1_N=2_t=0_U=2_V='+str(V)+'_J=0.h5'
        else:
                filename = 'obs/L=2_Ly=1_N=2_t=0_U=2_V=-6_J=0.h5'
        print(filename)
        try:
                f = h5py.File(filename,'r')
        except:
                print('file not there')
                continue
        Chis = []
        for Chi in np.array(list(map(int,list(f)))):
                Chis.append(int(Chi))
        Chis.sort()
        Chi = str(max(Chis))
        Chi = '550'
        print('V=',V,'Chi',Chi)
        tz = f[Chi+'/Tz'][0]
        tx = f[Chi+'/Tx'][0]
        t = np.sqrt(tz*tz+tx*tx)
        Ts[iV] = t
        
plt.plot(Vs, Ts, marker='.', label='$L=\infty$')

        # popt, pcov = curve_fit(second, Vs, Ts/L)
        # print(popt)
        # fits = np.zeros(len(Vs))
        # i=-1
        # for V in Vs:
        #         i=i+1
        #         fits[i] = second(V)
        # plt.plot(Vs, fits, 'g--')
# print(Es)
# for i in range(int(L/2)+1):
#     plt.plot(Vs, Es[i,:], marker='.', label='$T=$'+str(2*i+1))
plt.grid()
plt.xlabel('$V/t$')
plt.ylabel('$T/L$')
plt.legend()

# Tmin = 0
# Tmax = int(L/2)
# Vmin = -6.4
# Vmax = -5.0
# fig, ax2 = plt.subplots(1, 1)

# im = ax2.imshow(Es, interpolation='none', cmap=cm.gnuplot, norm=mcolors.SymLogNorm(1.e-10), aspect='auto', extent=[Vmin,Vmax,Tmax,Tmin])
# fig.colorbar(im)

if args.save:
    outfile = 'energy-of-V'+args.set+'_L='+str(L)
#    plt.savefig(outfile+'.svg')
    plt.savefig(outfile+'.pdf', bbox_inches='tight')
    os.system('pdfcrop '+outfile+'.pdf '+outfile+'.pdf')
plt.show()
