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
#parser.add_argument('-Nt', action='store', default=40)
parser.add_argument('-Nq', action='store', default=40)
parser.add_argument('-x0', action='store', default=20)
parser.add_argument('-Nw', action='store', default=1000)
parser.add_argument('-tmax', action='store', default=4)
parser.add_argument('-wmin', action='store', default=-4)
parser.add_argument('-wmax', action='store', default=2)
args = parser.parse_args()

wmin = float(args.wmin)
wmax = float(args.wmax)
tmax = int(args.tmax)
Nq = int(args.Nq)
Nt = int(tmax/0.1)
Nw = int(args.Nw)
x0 = int(args.x0)

filename = 'SigmawqIm_x0='+str(x0)+'_L='+str(2*x0)+\
                      '_tmax='+str(args.tmax)+'_Nt='+str(Nt)+\
                      '_qmin='+str(-1)+'_qmax='+str(1)+'_Nq='+str(Nq)+\
                      '_wmin='+str(args.wmin)+'_wmax='+str(args.wmax)+'_Nw='+str(Nw)+\
                      '.dat'
data = loadtxt(filename)
Nw = len(data[:,0])
# G: Nw x Nq
print('Nw=',Nw,'Nq=',Nq,'wmax=',wmax)

#norm=mcolors.LogNorm(vmin=1e-3, vmax=10), , 
im = imshow(abs(-1./pi*data), origin='lower', interpolation='none', cmap=cm.inferno, aspect='auto', extent=[-pi,pi,wmin,wmax])
colorbar()

maxs = []
for i in range(Nq):
	imax = argmax(-1./pi*data[:,i])
	maxs.append(linspace(wmin,wmax,Nw,endpoint=True)[imax])
plt.plot(linspace(-pi,pi,Nq,endpoint=True), maxs)

if args.save:
	
	print filename(-1,'.pdf')
	savefig(filename(-1,'.pdf'), bbox_inches='tight')
	os.system('pdfcrop '+filename(-1,'.pdf')+' '+filename(-1,'.pdf'))

plt.show()
