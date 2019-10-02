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
parser.add_argument('-spec', action='store', default='A1P')
args = parser.parse_args()

set = args.set
spec = args.spec
plot = args.plot
mu = 0

Lcell = 2
L = 20
tmax = 4
Nplots = 1
wmin = -10
wmax = +10

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
	res += '_t=1'
	res += '_U=0'
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
	res += '_t=1'
	res += '_U=0'
	res += '_L='+str(L)
	res += '_tmax='+str(tmax)
	res += '.h5'
	print(res)
	return res

qticks = [0, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 2*pi]
qlabels = ["$0$", "$\\frac{\pi}{2}$", "$\\frac{3\pi}{4}$", "$\pi$", "$\\frac{5\pi}{4}$", "$\\frac{3\pi}{2}$", "$2\pi$"]

if args.plot == 'specFull':
	
	fig, axs = plt.subplots(1,3)
	
	G1 = h5py.File(filename_wq(set,"PES",L,tmax),'r')
	G2 = h5py.File(filename_wq(set,"IPE",L,tmax),'r')
	G3 = h5py.File(filename_wq(set,"A1P",L,tmax),'r')
	
	datawq = -1./pi*(np.asarray(G1['G']['ωqIm'])+np.asarray(G2['G']['ωqIm']))
	
	im1 = axs[0].imshow(datawq, origin='lower', interpolation='none', cmap=cm.inferno, aspect='auto', extent=[0,2*pi,wmin-mu,wmax-mu])
	axs[0].set_ylabel('$\omega$')
	axs[0].grid()
	axs[0].set_xticks(qticks)
	axs[0].set_xticklabels(qlabels)
	fig.colorbar(im1, ax=axs[0])
	
	axs[0].plot(qvals,epsFvals, c='g')
	
	datawq = -1./pi*(np.asarray(G1['G00']['ωqIm'])+np.asarray(G2['G11']['ωqIm']))
	axs[1].imshow(datawq, origin='lower', interpolation='none', cmap=cm.inferno, aspect='auto', extent=[0,2*pi,wmin-mu,wmax-mu])
	
	axs[1].plot(qvals,eps0vals, c='c')
	axs[1].plot(qvals,eps1vals, c='b')
	
	dataQDOS = np.asarray(G3['G']['QDOS'])
	axs[2].plot(np.linspace(wmin,wmax,dataQDOS.shape[0],endpoint=True), dataQDOS)
	
#elif args.plot == 'QDOS':
#	
#	for i,VA in enumerate(VA_list):
#		for i,VB in enumerate(VB_list):
#		
#			A1Pwq = h5py.File(filename_wq(set,'A1P',L,tmax),'r')
##			PESwq = h5py.File(filename_wq(set,'PES',L,tmax),'r')
##			IPEwq = h5py.File(filename_wq(set,'IPE',L,tmax),'r')
##			SSFwq = h5py.File(filename_wq(set,'SSF',L,tmax),'r')
##			PEStx = h5py.File(filename_tx(set,'PES',L,tmax),'r')
##			IPEtx = h5py.File(filename_tx(set,'IPE',L,tmax),'r')
##			SSFtx = h5py.File(filename_tx(set,'SSF',L,tmax),'r')
#			
#			QDOS = A1Pwq['G']['QDOS']
#			waxis = linspace(wmin, wmax, QDOS.shape[0], endpoint=True)
#			qaxis = linspace(0, 2*pi, L/2+1, endpoint=True)
#			qaxis_ = linspace(0, 2*pi, L/2, endpoint=False)
##			mu = A1Pwq['μ'][0]
#			
#			plt.plot(waxis, QDOS, label='$U_A='+str(VA)+'$, $U_B='+str(VB)+'$')
#			plt.legend(fontsize=12)
#			plt.xlabel("$\omega$")
#			plt.ylabel("QDOS($\omega$)")

if args.save:
	
	savefig(args.plot+'.pdf', bbox_inches='tight')
	os.system('pdfcrop '+args.plot+'.pdf'+' '+args.plot+'.pdf')

plt.show()
