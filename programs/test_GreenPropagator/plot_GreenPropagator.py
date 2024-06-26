#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc
from matplotlib import rcParams
from numpy import amin, amax, savetxt, arange, argmax
from math import *
from scipy.optimize import curve_fit
import os, sys
import glob
import argparse
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties

#sys.path.insert(0, '../PYSNIP')
#import StringStuff

rc('text',usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('font',size=14)

parser = argparse.ArgumentParser()
parser.add_argument('-save', action='store_true', default=False)
parser.add_argument('-set', action='store', default='G')
parser.add_argument('-Nq', action='store', default=40)
parser.add_argument('-x0', action='store', default=20)
parser.add_argument('-Nw', action='store', default=1000)
parser.add_argument('-tmax', action='store', default=4)
parser.add_argument('-wmin', action='store', default=-10)
parser.add_argument('-wmax', action='store', default=10)
parser.add_argument('-i', action='store', default=0)
parser.add_argument('-j', action='store', default=0)
args = parser.parse_args()

wmin = float(args.wmin)
wmax = float(args.wmax)
tmax = int(args.tmax)
Nq = int(args.Nq)
Nt = int(tmax/0.1)
Nw = int(args.Nw)
x0 = int(args.x0)
if args.set == 'G':
	prefix = 'A1P_GωqIm'
elif args.set == 'PES':
	prefix = 'PES_GωqIm'
elif args.set == 'IPE':
	prefix = 'IPE_GωqIm'
elif args.set == 'Sigma':
	prefix = 'A1P_ΣωqIm'
i = int(args.i)
j = int(args.j)

qmin = 0
qmax = 2*pi

filenamePES = "PES_G=ωqIm_L=40_tmax=2_qmin=0_qmax=2_wmin=-10_wmax=10.dat"
filenameIPE = "IPE_G=ωqIm_L=40_tmax=2_qmin=0_qmax=2_wmin=-10_wmax=10.dat"
filenameA1P = "A1P_G=ωqIm_L=40_tmax=2_qmin=0_qmax=2_wmin=-10_wmax=10.dat"
dataPES = -1./pi*(loadtxt(filenamePES))
dataIPE = -1./pi*(loadtxt(filenameIPE))
dataSUM = dataPES + dataIPE
dataA1P = -1./pi*(loadtxt(filenameA1P))

filename00 = 'A1P_G=ωqIm_i=0_j=0_L=40_tmax=2_qmin=0_qmax=2_wmin=-10_wmax=10.dat'
filename11 = 'A1P_G=ωqIm_i=1_j=1_L=40_tmax=2_qmin=0_qmax=2_wmin=-10_wmax=10.dat'
#filename00 = 'A1P_G=ωqdiagIm_i=0_L=40_tmax=2_qmin=0_qmax=2_wmin=-10_wmax=10.dat'
#filename11 = 'A1P_G=ωqdiagIm_i=1_L=40_tmax=2_qmin=0_qmax=2_wmin=-10_wmax=10.dat'
data00 = -1./pi*(loadtxt(filename00))
data11 = -1./pi*(loadtxt(filename11))

filename_tx00 = 'A1P_G=txIm_i=0_j=0_L=40_tmax=2.dat'
filename_tx00good = './good_tPrime=0/A1P_G=txIm_i=0_j=0_L=40_tmax=2.dat'
data_tx00 = -1./pi*(loadtxt(filename_tx00))
data_tx00good = -1./pi*(loadtxt(filename_tx00good))

grid()

mu = 0

#ax1 = plt.subplot(131)
#ax2 = plt.subplot(132)
#ax3 = plt.subplot(133)
##abs(data_tx00[:,0:-1:2])
#im1 = ax1.imshow(abs(data_tx00), norm=mcolors.LogNorm(vmin=1e-7, vmax=1e1), origin='lower', interpolation='none', cmap=cm.rainbow, aspect='auto', extent=[0,39,0,2])
#ax1.set_title('current')
##colorbar(im1)
#im2 = ax2.imshow(abs(data_tx00good), norm=mcolors.LogNorm(vmin=1e-7, vmax=1e1), origin='lower', interpolation='none', cmap=cm.rainbow, aspect='auto', extent=[0,39,0,2])
#ax2.set_title('good')
#colorbar(im2)
##im3 = ax3.imshow(abs(data_tx00-data_tx00good), origin='lower', interpolation='none', cmap=cm.inferno, aspect='auto', extent=[0,39,0,2])
##ax3.set_title('diff')
##colorbar(im3)

ax2 = plt.subplot(111)
ax2.imshow(dataA1P, origin='lower', interpolation='none', cmap=cm.inferno, aspect='auto', extent=[qmin,qmax,wmin,wmax])
maxs = []
for i in range(len(dataA1P[0,:])):
	imax = argmax(dataA1P[:,i])
	maxs.append(linspace(wmin-mu,wmax-mu,Nw,endpoint=True)[imax])
plt.plot(linspace(0,2*pi,len(dataA1P[0,:]),endpoint=True), maxs, marker='.')

tA = 0
tB = 0
epsA = 0
epsB = 0
tperp = 1
tx = 1
qvals = linspace(qmin,qmax,31,endpoint=True)

def root(q):
	return sqrt( (0.5*(epsA-epsB)-(tA-tB)*cos(q))**2 + tperp**2+tx**2+2.*tperp*tx*cos(q) );

def eps0(q):
	return min(0.5*(epsA+epsB)-(tA+tB)*cos(q)+root(q), 0.5*(epsA+epsB)-(tA+tB)*cos(q)-root(q))

def eps1(q):
	return max(0.5*(epsA+epsB)-(tA+tB)*cos(q)+root(q), 0.5*(epsA+epsB)-(tA+tB)*cos(q)-root(q))

def eps(q):
	return -(tperp+tx)*cos(q)-(tA+tB)*cos(2*q)

eps0vals = []
eps1vals = []
eps_vals = []

for iq in range(31):
	eps0vals.append(eps0(qvals[iq]))
	eps1vals.append(eps1(qvals[iq]))
	eps_vals.append(eps(qvals[iq]))

ax2.plot(qvals,eps0vals)
ax2.plot(qvals,eps1vals)
ax2.plot(qvals,eps_vals)

if args.save:
	
	print filename(-1,'.pdf')
	savefig(filename(-1,'.pdf'), bbox_inches='tight')
	os.system('pdfcrop '+filename(-1,'.pdf')+' '+filename(-1,'.pdf'))

plt.show()
