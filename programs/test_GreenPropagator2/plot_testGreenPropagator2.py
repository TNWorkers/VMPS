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
parser.add_argument('-model', action='store', default='Hubbard')
args = parser.parse_args()

set = args.set
spec = args.spec
plot = args.plot

if args.model == 'Hubbard':
	Lcell = 2
	Ncells = 20
	tmax = 6
	U = 6
	sym = 'SU2⊗U1'
	INT = 'DIRECT' # INTERP || DIRECT
	L = Lcell*Ncells
	wmin = -10
	wmax = +10
	qmin = -1
	qmax = +1
	qticks = [-pi, -pi/2, 0, pi/2, pi]
	qlabels = ["$-\pi$", "$-\\frac{\pi}{2}$", "$0$", "$\\frac{\pi}{2}$", "$\pi$"]

elif args.model == 'Heisenberg':
	Lcell = 2
	Ncells = 20
	tmax = 6
	J = 1
	sym = 'SU2'
	INT = 'INTERP'
	L = Lcell*Ncells
	wmin = 0
	wmax = 10
	qmin = 0
	qmax = 2
	qticks = [0, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 2*pi]
	qlabels = ["$0$", "$\\frac{\pi}{2}$", "$\\frac{3\pi}{4}$", "$\pi$", "$\\frac{5\pi}{4}$", "$\\frac{3\pi}{2}$", "$2\pi$"]


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

def filename_wq(set,spec,L,tmax,qmin,qmax):
	res = set+'/'
	res += spec
	res += '_Lcell='+str(Lcell)
	res += '_sym='+str(sym)
	if args.model == 'Hubbard':
		res += '_U='+str(U)
	elif args.model == 'Heisenberg':
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
	res += '_Lcell='+str(Lcell)
	res += '_sym='+str(sym)
	if args.model == 'Hubbard':
		res += '_U='+str(U)
	elif args.model == 'Heisenberg':
		res += '_J='+str(J)
	res += '_L='+str(L)
	res += '_tmax='+str(tmax)
	res += '_INT='+INT
	res += '.h5'
	print(res)
	return res

if args.plot == 'specFull':
	
	fig, ax = plt.subplots(1,1)
	
	if args.model == 'Hubbard':
		G0 = h5py.File(filename_wq(".","PES",L,tmax,qmin,qmax),'r')
		G1 = h5py.File(filename_wq(".","IPE",L,tmax,qmin,qmax),'r')
		datawq = -1./pi * np.asarray(G0['G']['ωqIm']) -1./pi * np.asarray(G1['G']['ωqIm'])
		print(G0['G'].keys()) #ωqIm
	elif args.model == 'Heisenberg':
		G0 = h5py.File(filename_wq(".","SSF",L,tmax,qmin,qmax),'r')
		datawq = -1./pi * np.asarray(G0['G']['ωqIm'])
	
	im1 = imshow(datawq, origin='lower', interpolation='none', cmap=cm.terrain, aspect='auto', extent=[qmin*pi,qmax*pi,wmin,wmax])
	ax.set_ylabel('$\omega$')
	ax.grid()
	ax.set_xticks(qticks)
	ax.set_xticklabels(qlabels)
	fig.colorbar(im1, ax=ax)
	
	Nw = datawq.shape[0]
	Nq = datawq.shape[1]
	qaxis = linspace(0, 2*pi, Nq, endpoint=True)
	waxis = linspace(wmin, wmax, Nw, endpoint=True)
	
	#plt.plot(qaxis, -2.*np.cos(np.asarray(qaxis)), c='r')

elif args.plot == 'QDOS':
	
	fig, ax = plt.subplots(1,1)
	
	if args.model == 'Hubbard':
		G0 = h5py.File(filename_wq(".","PES",L,tmax,qmin,qmax),'r')
		G1 = h5py.File(filename_wq(".","IPE",L,tmax,qmin,qmax),'r')
		dataw = np.asarray(G0['G']['QDOS']) + np.asarray(G1['G']['QDOS'])
	elif args.model == 'Heisenberg':
		G0 = h5py.File(filename_wq(".","SSF",L,tmax,qmin,qmax),'r')
		dataw = np.asarray(G0['G']['QDOS'])
	
	Nw = dataw.shape[0]
	waxis = linspace(wmin, wmax, Nw, endpoint=True)
	
	plt.plot(waxis, dataw)
	plt.grid()

if args.save:
	
	savefig(args.plot+'.pdf', bbox_inches='tight')
	os.system('pdfcrop '+args.plot+'.pdf'+' '+args.plot+'.pdf')

plt.show()
