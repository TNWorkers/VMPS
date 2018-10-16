#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc
from matplotlib import rcParams
from numpy import amin, amax, savetxt
from scipy.optimize import curve_fit
import os, sys
import glob
import argparse
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties

def round(x):
	if x==int(x):
		return int(x)
	else:
		return x

rc('text',usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('font',size=14)

dE_list = [0.5, 0.4, 0.3, 0.25, 0.2]
dE_listring = ','.join([str(dE) for dE in dE_list])

parser = argparse.ArgumentParser()
parser.add_argument('-save', action='store_true', default=False)
parser.add_argument('-redraw', action='store_true', default=False)
parser.add_argument('-spec', action='store', default='IPES')
parser.add_argument('-sym', action='store', default='U1')
parser.add_argument('-L', action='store', default=20)
parser.add_argument('-M', action='store', default=20)
parser.add_argument('-N', action='store', default=0)
parser.add_argument('-J', action='store', default=9)
parser.add_argument('-U', action='store', default=0)
parser.add_argument('-sigma', action='store', default="↓")
parser.add_argument('-dE', action='store', default=0.2)
args = parser.parse_args()

sym = args.sym
spec = args.spec
L = int(args.L)
M = int(args.M)
N = int(args.N)
J = round(float(args.J))
U = round(float(args.U))
sigma = args.sigma
dE = float(args.dE)

def filename(i0, ext='.dat'):
	out = spec+'_L='+str(L)+'_M='+str(M)+'_N='+str(N)+'_J='+str(J)+'_U='+str(U)
	if spec == 'PES' or spec == 'IPES':
		out += '_sigma='+sigma
	if i0 != -1:
		out += '_i0='+str(i0)
	out += '_dE='+str(dE)
	out += ext
	return out

specfile = './'+sym+'/data/kspec_'+filename(-1)
Efile = './'+sym+'/data/Eminmax_'+filename(-1)
wd = './'+sym+'/'+spec+'/'

dataset = loadtxt(wd+filename(0))
Nrows = len(dataset[:,0])

Eval = []
kval = []
G2d = []
Eminmax = []

def trigfac (i,j,kval):
	if i == j:
		return 2.
	else:
		return 2.*cos(kval*float(i-j))

if os.path.isfile(specfile) and os.path.isfile(Efile) and not args.redraw:
	
	G2d = loadtxt(specfile)
	Eminmax = loadtxt(Efile)
	
else:
	
	for ik in np.arange(-L/2., L/2.+1., 1.):
		
		k = 2.*pi/L*ik
		print 'ik=',int(ik),'k=',k,'k/π=',k/pi
		
		kspec = [0+0j]*Nrows
		
		for j in range(0,L/2+1):
			
			dataset = loadtxt(wd+filename(j))
			
			if len(Eminmax)!=2:
				Eminmax.append(dataset[ 0,0])
				Eminmax.append(dataset[-1,0])
			
			for i in range(0,L-1):
				
				kspec += trigfac(i,j, k) * dataset[:,1+i]
				
		G2d.append(abs(kspec.real)/L)
	
	savetxt(specfile,G2d)
	savetxt(Efile,Eminmax)

kmin, kmax = -pi, pi
maxval = 1e1
minval = 1e-3

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

#CMRmap
#norm=mcolors.LogNorm(vmin=minval, vmax=maxval), 
im = ax2.imshow(G2d, interpolation='none', cmap=cm.gnuplot, aspect='auto', extent=[Eminmax[0],Eminmax[1],kmin,kmax])

plt.xlabel('$\omega$', fontsize=16)
plt.ylabel('$k$', fontsize=16)
plt.yticks([-pi, -pi/2, 0, pi/2, pi], ['$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'+$\frac{\pi}{2}$', r'+$\pi$'])
ax2.tick_params(axis='x', which='both', direction='in', colors='white', grid_color='white', labelcolor='k', length=6)

dataset = loadtxt(wd+filename(L/2))
ax1.set_xlim(Eminmax[0], Eminmax[1])
ax1.plot(dataset[:,0], dataset[:,L/2+1])

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

if args.save:
	
#	savefig(filename(-1,'.png'), dpi=100, bbox_inches='tight')
#	savefig(filename(-1,'.eps'), bbox_inches='tight')
	savefig(filename(-1,'.pdf'), bbox_inches='tight')
	system('pdfcrop '+outfile+'.pdf '+outfile+'.pdf')

plt.show()
