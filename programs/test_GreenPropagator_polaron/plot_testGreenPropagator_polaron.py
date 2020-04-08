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
parser.add_argument('-spec', action='store', default='IPE')
parser.add_argument('-sym', action='store', default='SU2')
parser.add_argument('-INT', action='store', default='DIRECT') # INTERP || DIRECT
args = parser.parse_args()

set = args.set
spec = args.spec
plot = args.plot
INT = args.INT

model = 'Kondo'
L = 20
tmax = 6
J = 12
if args.sym == 'U1':
	sym = 'U1⊗U1'
elif args.sym == 'SU2':
	sym = 'SU2⊗U1'
wmin = -15
wmax = +15
qmin = -1
qmax = +1
qticks = [-pi, -pi/2, 0, pi/2, pi]
qlabels = ["$-\pi$", "$-\\frac{\pi}{2}$", "$0$", "$\\frac{\pi}{2}$", "$\pi$"]

def filename_wq(set,spec,L,tmax,qmin,qmax):
	res = set+'/'
	res += spec
	res += '_L='+str(L)
	res += '_model='+model
	res += '_sym='+str(sym)
	res += '_J='+str(J)
	res += '_L='+str(L)
	res += '_tmax='+str(tmax)
	res += '_INT='+INT
	res += '_qmin='+str(qmin)
	res += '_qmax='+str(qmax)
	res += '_wmin='+str(wmin)+'_wmax='+str(wmax)
	res += '.h5'
	print(res)
	return res

def filename_tx(set,spec,L,tmax,qmin,qmax):
	res = set+'/'
	res += spec
	res += '_L='+str(L)
	res += '_model='+model
	res += '_sym='+str(sym)
	res += '_J='+str(J)
	res += '_L='+str(L)
	res += '_tmax='+str(tmax)
	res += '_INT='+INT
	res += '.h5'
	print(res)
	return res

if args.plot == 'specFull':
	
	fig, (ax1, ax2) = plt.subplots(1,2)
	
	if sym == 'U1⊗U1':
		G = h5py.File(filename_wq(".",'IPEDN',L,tmax,qmin,qmax),'r')
		datawq1 = -1./pi * np.asarray(G['G']['ωqIm'])
		G = h5py.File(filename_wq(".",'IPE',L,tmax,qmin,qmax),'r')
		datawq2 = -1./pi * np.asarray(G['G']['ωqIm'])
	elif sym == 'SU2⊗U1':
		G = h5py.File(filename_wq(".",spec,L,tmax,qmin,qmax),'r')
		datawq1 = -1./pi * np.asarray(G['G.Q0']['ωqIm']) # spin-dn
		datawq2 = -1./pi * np.asarray(G['G.Q1']['ωqIm']) # spin-up
	
	im1 = ax1.imshow(datawq1, origin='lower', interpolation='none', cmap=cm.terrain, aspect='auto', extent=[qmin*pi,qmax*pi,wmin,wmax])
	ax1.set_ylabel('$\omega$')
	ax1.grid()
	ax1.set_xticks(qticks)
	ax1.set_xticklabels(qlabels)
	fig.colorbar(im1, ax=ax1)
	
	im2 = ax2.imshow(datawq2, origin='lower', interpolation='none', cmap=cm.terrain, aspect='auto', extent=[qmin*pi,qmax*pi,wmin,wmax])
	ax2.set_ylabel('$\omega$')
	ax2.grid()
	ax2.set_xticks(qticks)
	ax2.set_xticklabels(qlabels)
	fig.colorbar(im2, ax=ax2)
	
	Nw = datawq1.shape[0]
	Nq = datawq1.shape[1]
	qaxis = linspace(0, 2*pi, Nq, endpoint=True)
	waxis = linspace(wmin, wmax, Nw, endpoint=True)
	
	#plt.plot(qaxis, -2.*np.cos(np.asarray(qaxis)), c='r')

elif args.plot == 'QDOS':
	
	fig, ax = plt.subplots(1,1)
	
	G0 = h5py.File(filename_wq(".",spec,L,tmax,qmin,qmax),'r')
	dataw = np.asarray(G0['G']['QDOS'])
	
	Nw = dataw.shape[0]
	waxis = linspace(wmin, wmax, Nw, endpoint=True)
	
	plt.plot(waxis, dataw)
	plt.grid()

if args.save:
	
	savefig(args.plot+'.pdf', bbox_inches='tight')
	os.system('pdfcrop '+args.plot+'.pdf'+' '+args.plot+'.pdf')

plt.show()
