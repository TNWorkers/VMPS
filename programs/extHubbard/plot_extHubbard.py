#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function #python2 is then compatible with python3 print syntax. The other way around is not possible.

import os, sys
import numpy as np
#from termcolor import colored, cprint
import matplotlib.pyplot as plt
from math import *
from cmath import *
from matplotlib import rc, rcParams
import h5py
from matplotlib.colors import LogNorm

sys.path.insert(0, '../PYSNIP')
#import folders
# import StringStuff

def myRound(x):
    if x==0.:
        return 0
    elif x==1.:
        return 1
    elif x==2.:
        return 2
    elif x==3.:
        return 3
    elif x==4.:
        return 4
    elif x==int(x):
        return int(x)
    else:
        return x

rc('text', usetex=True)
rc('font', **{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('font', size=12)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-J', action='store', default=0)
parser.add_argument('-save', action='store', default=False)
parser.add_argument('-set', action='store', default='001')
args = parser.parse_args()

def obsFilename(U,V,J):
    return "U="+str(U)+"_V="+str(abs(V))+"_J="+str(J)

Vs = np.arange(-5,5+1,1)
Us = np.arange(0,8+1,1)

Tmax    = [[0 for x in range(len(Vs))] for y in range(len(Us))]
Smax    = [[0 for x in range(len(Vs))] for y in range(len(Us))]
spinon  = [[0 for x in range(len(Vs))] for y in range(len(Us))]
holon   = [[0 for x in range(len(Vs))] for y in range(len(Us))]
entropy = [[0 for x in range(len(Vs))] for y in range(len(Us))]
error   = [[0 for x in range(len(Vs))] for y in range(len(Us))]

for iU in range(len(Us)):
    for iV in range(len(Vs)):
        
        U = Us[iU]
        V = Vs[iV]
        J = args.J
        
        filename = './PROG/cluster-calcHUBB/g++-U'+str(U)+'.0-V'+str(V)+'.0-J'+str(J)+'.0-L2-Ly1-a32ea3e6-'+str(args.set)+'/obs/'+obsFilename(U,V,J)+'.h5'
        
        if os.path.isfile(filename):
            f = h5py.File(filename,'r')
            
            Chis = []
            logChis = []
            S0 = []
            S1 = []
            
            for Chi in np.array(list(map(int,list(f)))):
                Chis.append(int(Chi))
            Chis.sort()
            
            for Chi in Chis:
                S0.append(f[str(Chi)]['Entropy'][0][0])
                S1.append(f[str(Chi)]['Entropy'][1][0])
                logChis.append(log(Chi))
            
            #fit0 = np.poly1d(np.polyfit(logChis, S0, 1))
            #fit1 = np.poly1d(np.polyfit(logChis, S1, 1))
            #print('fit S=c/3*log(χ)+γ:')
            #print('c=',np.real(fit0(0))/3, 'γ=',np.real(fit0(1)))
            #print('c=',np.real(fit1(0))/3, 'γ=',np.real(fit1(1)))
            
            #plt.xlabel('$\ln \chi$')
            #plt.ylabel('$S$')
            #plt.plot(logChis, S0, marker='.', label='S0')
            #plt.plot(logChis, S1, marker='.', label='S1')
            #plt.plot(logChis, fit0(logChis), marker='.', label='fit0')
            #plt.plot(logChis, fit1(logChis), marker='.', label='fit1')
            #plt.legend()
            
            print("U=",U,"V=",V)
            print('χs=',Chis)
            Chi = str(max(Chis))
            print('using χ=', Chi)
            
            #print("Keys: %s" % f[Chi].keys())
            print('Dmax=', f[Chi]['Dmax'][0])
            print('Mmax=', f[Chi]['Mmax'][0])
            print('err_eigval=', f[Chi]['err_eigval'][0])
            print('err_state=', f[Chi]['err_state'][0])
            print('err_var=', f[Chi]['err_var'][0])
            print('S=', f[Chi]['Entropy'][0][0], f[Chi]['Entropy'][1][0])
            print('nh=', f[Chi]['nh'][0][0], f[Chi]['nh'][1][0])
            print('ns=', f[Chi]['ns'][0][0], f[Chi]['ns'][1][0])
            
            Smax[iU][iV] = f[Chi]['S_pi'][0][0] #max(setk) #
            Tmax[iU][iV] = f[Chi]['T_pi'][0][0] #max(setk) #
            spinon[iU][iV] = f[Chi]['ns'][0][0] 
            holon[iU][iV] = f[Chi]['nh'][0][0]
            entropy[iU][iV] = f[Chi]['Entropy'][0][0]
            error[iU][iV] =  f[Chi]['err_state'][0]
            print("")
        else:
            print("no file for U=",U,"V=",V)
            print("")

fig = plt.figure()

cmap = 'rainbow'

ax1 = fig.add_subplot(231)
im1 = ax1.imshow(Smax, interpolation='nearest', origin='lower', cmap=cmap, extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax1.set_title('spin S')
ax1.set_ylabel('U', fontsize=14)
fig.colorbar(im1)

ax2 = fig.add_subplot(232)
im2 = ax2.imshow(Tmax, interpolation='nearest', origin='lower', cmap=cmap, extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax2.set_title('pseudo-spin T')
fig.colorbar(im2)

ax3 = fig.add_subplot(233)
im3 = ax3.imshow(entropy, interpolation='nearest', origin='lower', cmap=cmap, extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax3.set_title('entropy')
fig.colorbar(im3)

ax4 = fig.add_subplot(234)
im4 = ax4.imshow(spinon, interpolation='nearest', origin='lower', vmin=0, vmax=1, cmap='seismic', extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax4.set_title('spinon')
fig.colorbar(im4)
ax4.set_xlabel('V', fontsize=14)

ax5 = fig.add_subplot(235)
im5 = ax5.imshow(holon, interpolation='nearest', origin='lower', vmin=0, vmax=1, cmap='seismic', extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax5.set_title('holon')
fig.colorbar(im5)

ax6 = fig.add_subplot(236)
im6 = ax6.imshow(error, interpolation='nearest', origin='lower', norm=LogNorm(), cmap=cmap, extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax6.set_title('state error')
fig.colorbar(im6)

if args.save:
    outfile = 'PhaseDiagram'
    plt.savefig(outfile+'.pdf', bbox_inches='tight')
    os.system('pdfcrop '+outfile+'.pdf '+outfile+'.pdf')

plt.show()