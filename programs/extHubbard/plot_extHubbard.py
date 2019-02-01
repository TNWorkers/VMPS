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
from matplotlib.colors import LogNorm

#sys.path.insert(0, '../PYSNIP')
#import folders
# import StringStuff

#def myRound(x):
#    if x==0.:
#        return 0
#    elif x==1.:
#        return 1
#    elif x==2.:
#        return 2
#    elif x==3.:
#        return 3
#    elif x==4.:
#        return 4
#    elif x==int(x):
#        return int(x)
#    else:
#        return x

#def round(x):
#	if x==int(x):
#		return int(x)
#	else:
#		return x

rc('text', usetex=True)
rc('font', **{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('font', size=12)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-J', action='store', default=0.5)
parser.add_argument('-save', action='store', default=False)
parser.add_argument('-set', action='store', default='004')
args = parser.parse_args()

def folderName(U,V,J):
    return '../../../cluster-calcHUBB/g++-U'+str(U)+'.0-V'+str(V)+'.0-J'+str(J)+'-L2-Ly1-a32ea3e6-'+str(args.set)

def obsFilename(U,V,J):
    return "U="+str(U)+"_V="+str(V)+"_J="+str(J)+'.h5'

Vs = np.arange(-5,5+1,1)
Us = np.arange(0,8+1,1)

T1pi    = [[0 for x in range(len(Vs))] for y in range(len(Us))]
S1pi    = [[0 for x in range(len(Vs))] for y in range(len(Us))]
T0pi    = [[0 for x in range(len(Vs))] for y in range(len(Us))]
S0pi    = [[0 for x in range(len(Vs))] for y in range(len(Us))]
spinon  = [[0 for x in range(len(Vs))] for y in range(len(Us))]
holon   = [[0 for x in range(len(Vs))] for y in range(len(Us))]
entropy = [[0 for x in range(len(Vs))] for y in range(len(Us))]
error   = [[-1 for x in range(len(Vs))] for y in range(len(Us))]
PD      = [[0 for x in range(len(Vs))] for y in range(len(Us))]

for iU in range(len(Us)):
    for iV in range(len(Vs)):
        
        U = Us[iU]
        V = Vs[iV]
        J = args.J
        
        filename = folderName(U,V,J)+'/obs/'+obsFilename(U,V,J)
        
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
            if len(Chis)>0:
                Chi = str(max(Chis))
                print('using χ=', Chi)
                
                print("Keys: %s" % f[Chi].keys())
                print('Dmax=', f[Chi]['Dmax'][0])
                print('Mmax=', f[Chi]['Mmax'][0])
                print('err_eigval=', f[Chi]['err_eigval'][0])
                print('err_state=', f[Chi]['err_state'][0])
                print('err_var=', f[Chi]['err_var'][0])
                print('entropy=', f[Chi]['Entropy'][0][0], f[Chi]['Entropy'][1][0])
                print('nh=', f[Chi]['nh'][0][0], f[Chi]['nh'][1][0])
                print('ns=', f[Chi]['ns'][0][0], f[Chi]['ns'][1][0])
                print('nh+ns-1=', f[Chi]['nh'][0][0]+f[Chi]['ns'][0][0]-1, f[Chi]['nh'][1][0]+f[Chi]['ns'][1][0]-1)
                print('S(π)=', f[Chi]['S_pi'][0], 'S(0)=', f[Chi]['S_0'][0])
                print('S(π)=', f[Chi]['S_pi'][1], 'S(0)=', f[Chi]['S_0'][1])
                print('T(π)=', f[Chi]['T_pi'][0], 'T(0)=', f[Chi]['T_0'][0])
                print('T(π)=', f[Chi]['T_pi'][1], 'T(0)=', f[Chi]['T_0'][1])
                
                S1pi[iU][iV] = f[Chi]['S_pi'][0][0]-f[Chi]['S_pi'][0][1]
                T1pi[iU][iV] = f[Chi]['T_pi'][0][0]-f[Chi]['T_pi'][0][1]
                S0pi[iU][iV] = f[Chi]['S_0'][0][0]+f[Chi]['S_0'][0][1]
                T0pi[iU][iV] = f[Chi]['T_0'][0][0]+f[Chi]['S_0'][0][1]
                spinon[iU][iV] = f[Chi]['ns'][0][0] 
                holon[iU][iV] = f[Chi]['nh'][0][0]
                entropy[iU][iV] = f[Chi]['Entropy'][0][0]
                error[iU][iV] =  f[Chi]['err_state'][0]
                
                if   max(S1pi[iU][iV], T1pi[iU][iV], T0pi[iU][iV], S0pi[iU][iV]) == S0pi[iU][iV]:
                    PD[iU][iV] = 1.5
                elif max(S1pi[iU][iV], T1pi[iU][iV], T0pi[iU][iV], S0pi[iU][iV]) == S1pi[iU][iV]:
                    PD[iU][iV] = 2.5
                elif max(S1pi[iU][iV], T1pi[iU][iV], T0pi[iU][iV], S0pi[iU][iV]) == T0pi[iU][iV]:
                    PD[iU][iV] = 3.5
                elif max(S1pi[iU][iV], T1pi[iU][iV], T0pi[iU][iV], S0pi[iU][iV]) == T1pi[iU][iV]:
                    PD[iU][iV] = 4.5
                print("")
        else:
            print("no file for U=",U,"V=",V)
            print("")

fig = plt.figure()

cmap = 'gnuplot'
title_fontsize = 12
label_fontsize = 12

#######

ax1 = fig.add_subplot(331)
im1 = ax1.imshow(S1pi, interpolation='nearest', origin='lower', cmap=cmap, extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax1.set_title('$S(k=\pi)$', fontsize=title_fontsize)
ax1.set_ylabel('U', fontsize=label_fontsize)
fig.colorbar(im1)
ax1.xaxis.set_major_formatter(plt.NullFormatter())

ax2 = fig.add_subplot(332)
im2 = ax2.imshow(T1pi, interpolation='nearest', origin='lower', cmap=cmap, extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax2.set_title('$T(k=\pi)$', fontsize=title_fontsize)
fig.colorbar(im2)
ax2.xaxis.set_major_formatter(plt.NullFormatter())
ax2.yaxis.set_major_formatter(plt.NullFormatter())

PD_colors = ['black', 'blue', 'red', 'yellow', 'green']
discrete = colors.ListedColormap(PD_colors)
bounds=[0,1,2,3,4,5]
discrete_norm = colors.BoundaryNorm(bounds,len(PD_colors))
ax3 = fig.add_subplot(333)
im3 = ax3.imshow(PD, interpolation='nearest', origin='lower', cmap=discrete, norm=discrete_norm, extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax3.set_title('phase diagram', fontsize=title_fontsize)
PD_colorbar = fig.colorbar(im3, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
PD_colorbar.ax.set_yticklabels(['unfinished','FM', 'AFM', 's-wave', '$\eta$-wave'])
ax3.xaxis.set_major_formatter(plt.NullFormatter())
ax3.yaxis.set_major_formatter(plt.NullFormatter())

#######

ax4 = fig.add_subplot(334)
im4 = ax4.imshow(S0pi, interpolation='nearest', origin='lower', cmap=cmap, extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax4.set_title('$S(k=0)$', fontsize=title_fontsize)
ax4.set_ylabel('U', fontsize=label_fontsize)
fig.colorbar(im4)
ax4.xaxis.set_major_formatter(plt.NullFormatter())

ax5 = fig.add_subplot(335)
im5 = ax5.imshow(T0pi, interpolation='nearest', origin='lower', cmap=cmap, extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax5.set_title('$T(k=0)$', fontsize=title_fontsize)
fig.colorbar(im5)
ax5.xaxis.set_major_formatter(plt.NullFormatter())
ax5.yaxis.set_major_formatter(plt.NullFormatter())

ax6 = fig.add_subplot(336)
im6 = ax6.imshow(error, interpolation='nearest', origin='lower', norm=LogNorm(), cmap='jet', vmin=1e-4, vmax=1, extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax6.set_title('state error', fontsize=title_fontsize)
fig.colorbar(im6)
ax6.xaxis.set_major_formatter(plt.NullFormatter())
ax6.yaxis.set_major_formatter(plt.NullFormatter())

#######

ax7 = fig.add_subplot(337)
im7 = ax7.imshow(holon, interpolation='nearest', origin='lower', cmap='seismic', vmin=0, vmax=1, extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax7.set_title('$\left<n_h\\right>$', fontsize=title_fontsize)
ax7.set_ylabel('U', fontsize=label_fontsize)
ax7.set_xlabel('V', fontsize=label_fontsize)
fig.colorbar(im7)

ax8 = fig.add_subplot(338)
im8 = ax8.imshow(spinon, interpolation='nearest', origin='lower', cmap='seismic', vmin=0, vmax=1, extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax8.set_title('$\left<n_s\\right>$', fontsize=title_fontsize)
ax8.set_xlabel('V', fontsize=label_fontsize)
fig.colorbar(im8)
ax8.yaxis.set_major_formatter(plt.NullFormatter())

ax9 = fig.add_subplot(339)
im9 = ax9.imshow(entropy, interpolation='nearest', origin='lower', norm=LogNorm(), cmap='jet', vmin=1, vmax=5, extent=[Vs[0],Vs[-1],Us[0],Us[-1]])
ax9.set_title('entropy', fontsize=title_fontsize)
ax9.set_xlabel('V', fontsize=12)
fig.colorbar(im9)
ax9.yaxis.set_major_formatter(plt.NullFormatter())

#######

if args.save:
    outfile = 'UVJ'+args.set
    plt.savefig(outfile+'.pdf', bbox_inches='tight')
    os.system('pdfcrop '+outfile+'.pdf '+outfile+'.pdf')

plt.show()
