#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc
from matplotlib import rcParams
from numpy import amin, amax, savetxt, arange, argmax, linspace, linalg
from scipy.optimize import curve_fit
import os, sys
import glob
import argparse
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from numpy import linalg as LA
import h5py
from numpy.linalg import inv

#sys.path.insert(0, '../PYSNIP')
#import StringStuff

rc('text',usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('font',size=12)

parser = argparse.ArgumentParser()
parser.add_argument('-save', action='store_true', default=False)
parser.add_argument('-set', action='store', default='.')
parser.add_argument('-plot', action='store', default='specFull')
parser.add_argument('-spec', action='store', default='SSF')
args = parser.parse_args()

set = args.set
spec = args.spec
plot = args.plot
mu = 0

Lcell = 2
Ncells = 20
tmax = 4
Nplots = 1
L = Lcell*Ncells
wmin = -5
wmax = +5

def root(q):
	return sqrt(2.+2.*cos(q));

def eps0(q):
	return min(+root(q),-root(q))

def eps1(q):
	return max(+root(q),-root(q))

def epsF(q):
	return -2.*cos(q)

eps0vals = []
eps1vals = []
epsFvals = []

for iq in range(101):
	qvals = np.asarray(linspace(0,2*pi,101,endpoint=True))
	eps0vals.append(eps0(qvals[iq]))
	eps1vals.append(eps1(qvals[iq]))
	epsFvals.append(epsF(qvals[iq]))

def filename_wq(set,spec,L,tmax):
	res = set+'/'
	res += spec
	res += '_Lcell='+str(Lcell)
	res += '_J=0'
	res += '_L='+str(L)
	res += '_tmax='+str(tmax)
	res += '_qmin=0_qmax=2'
	res += '_wmin='+str(wmin)+'_wmax='+str(wmax)
	res += '.h5'
	print(res)
	return res

def filename_tx(set,spec,L,tmax):
	res = set+'/'
	res += spec
	res += '_Lcell='+str(Lcell)
	res += '_U=0'
	res += '_L='+str(L)
	res += '_tmax='+str(tmax)
	res += '.h5'
	print(res)
	return res

qticks = [0, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 2*pi]
qlabels = ["$0$", "$\\frac{\pi}{2}$", "$\\frac{3\pi}{4}$", "$\pi$", "$\\frac{5\pi}{4}$", "$\\frac{3\pi}{2}$", "$2\pi$"]

if args.plot == 'specFull':
	
	fig, ax = plt.subplots(1,1)
	
	G0 = h5py.File(filename_wq(set,spec,L,tmax),'r')
	datawq = -1./pi * np.asarray(G0['G']['ωqIm'])
	
	im1 = imshow(datawq, origin='lower', interpolation='none', cmap=cm.terrain, aspect='auto', extent=[0,2*pi,wmin-mu,wmax-mu])
	ax.set_ylabel('$\omega$')
	ax.grid()
	ax.set_xticks(qticks)
	ax.set_xticklabels(qlabels)
	fig.colorbar(im1, ax=ax)
	
	Nw = datawq.shape[0]
	Nq = datawq.shape[1]
	qaxis = linspace(0, 2*pi, Nq, endpoint=True)
	waxis = linspace(wmin, wmax, Nw, endpoint=True)
	
	plt.plot(qaxis, -2.*np.cos(np.asarray(qaxis)), c='r')

if args.save:
	
	savefig(args.plot+'.pdf', bbox_inches='tight')
	os.system('pdfcrop '+args.plot+'.pdf'+' '+args.plot+'.pdf')

plt.show()
