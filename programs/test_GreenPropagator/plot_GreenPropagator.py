#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc
from matplotlib import rcParams
from numpy import amin, amax, savetxt, arange, argmax
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
parser.add_argument('-wmin', action='store', default=-5)
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

#filename = prefix+'_i='+str(i)+'_j='+str(j)+\
#                  '_x0='+str(x0)+'_L='+str(2*x0)+\
#                  '_tmax='+str(args.tmax)+'_Nt='+str(Nt)+\
#                  '_qmin='+str(-1)+'_qmax='+str(1)+'_Nq='+str(Nq)+\
#                  '_wmin='+str(args.wmin)+'_wmax='+str(args.wmax)+'_Nw='+str(Nw)+\
#                  '.dat'
filename = 'PES_GωqIm_x0=20_L=40_tmax=12_Nt=60_qmin=-1_qmax=1_Nq=41_wmin=-5_wmax=10_Nw=1000.dat'
data = loadtxt(filename)

grid()

mu = 3
if args.set == 'Sigma':
	im = imshow(abs(data), origin='lower', norm=mcolors.LogNorm(vmin=1e-2, vmax=1e2), interpolation='none', cmap=cm.inferno, aspect='auto', extent=[-pi,pi,wmin-mu,wmax-mu])
else:
	im = imshow(abs(-1./pi*data), origin='lower', norm=mcolors.LogNorm(vmin=1e-2, vmax=1e0), interpolation='none', cmap=cm.inferno, aspect='auto', extent=[-pi,pi,wmin-mu,wmax-mu])
colorbar()

#maxs = []
#for i in range(Nq):
#	imax = argmax(-1./pi*data[:,i])
#	maxs.append(linspace(wmin-mu,wmax-mu,Nw,endpoint=True)[imax])
#plt.plot(linspace(-pi,pi,Nq,endpoint=True), maxs)

if args.save:
	
	print filename(-1,'.pdf')
	savefig(filename(-1,'.pdf'), bbox_inches='tight')
	os.system('pdfcrop '+filename(-1,'.pdf')+' '+filename(-1,'.pdf'))

plt.show()
