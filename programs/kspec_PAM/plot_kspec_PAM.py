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
from matplotlib.colors import LogNorm
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

#sys.path.insert(0, '../PYSNIP')
#import StringStuff

rc('text',usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('font',size=10)
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

parser = argparse.ArgumentParser()
parser.add_argument('-save', action='store_true', default=False)
parser.add_argument('-set', action='store', default='.')
parser.add_argument('-plot', action='store', default='cell')
parser.add_argument('-spec', action='store', default='A1P')
parser.add_argument('-L', action='store', type=int, default=32)
parser.add_argument('-U', action='store', type=int, default=0)
parser.add_argument('-Ly', action='store', type=int, default=1)
parser.add_argument('-tol', action='store', type=float, default=0.01)
parser.add_argument('-dt', action='store', type=float, default=0.1)
parser.add_argument('-tmax', action='store', type=int, default=4)
parser.add_argument('-INT', action='store', default='OOURA')
parser.add_argument('-index', action='store', type=int, default=0)
args = parser.parse_args()

set = args.set
spec = args.spec
plot = args.plot
INT = args.INT
index = args.index

U = args.U
tfc = 1
tcc = 0
tff = 0
Retx = 1
Imtx = 0
Rety = 0
Imty = 0

dt = args.dt
tolDeltaS = args.tol
tmax = args.tmax

L = args.L

long_spec = {'A1P':'one-particle', 'PES':'photoemission', 'IPE':'inv. photoemission', 'SSF':'spin', 'PSZ':'charge'}

def calc_ylim(UA,spec):
	ylim0 = {'A1P':-10, 'PES':-10, 'IPE':-10}
	ylim1 = {'A1P':+10, 'PES':+10, 'IPE':+10}
	return ylim0[spec], ylim1[spec]

wmin = -10
wmax = +10

qticks = [0, pi/2, pi, 3*pi/2, 2*pi]
qlabels = ['$0$', '$\\frac{\pi}{2}$', '$\pi$', '$\\frac{3\pi}{2}$', '$2\pi$']

def filename_wq(set,spec,L,tfc,tcc,tff,Retx,Imtx,Rety,Imty,U,tmax,wmin,wmax):
	res = set+'/'
	res += spec
#	PES_L=4_N=4_U=0_tfc=1_tcc=1_tff=0_tx=01_ty=00_L=32_dLphys=2_tmax=4_INT=OOURA_qmin=0_qmax=2_wmin=-10_wmax=10
	res += '_tfc='+str(tfc)
	res += '_tcc='+str(tcc)
	res += '_tff='+str(tff)
	res += '_tx='+str(Retx)+","+str(Imtx)
	res += '_ty='+str(Rety)+","+str(Imty)
	res += '_U='+str(U)
#	res += '_dt='+str(dt)
#	res += '_tolΔS='+str(tolDeltaS)
	res += '_L='+str(L)
#	res += '_dLphys='+str(2)
	res += '_tmax='+str(tmax)
	res += '_INT='+INT
	res += '_qmin=0_qmax=2'
	res += '_wmin='+str(wmin)+'_wmax='+str(wmax)
	res += '.h5'
	print(res)
	return res

def open_Gwq (spec,index):
	
	Gstr = 'G'+str(index)+str(index)
	
	if spec == 'A1P':
		G = h5py.File(filename_wq(set,'PES',L,tfc,tcc,tff,Retx,Imtx,Rety,Imty,U,tmax,wmin,wmax),'r')
	else:
		G = h5py.File(filename_wq(set,spec,L,tfc,tcc,tff,Retx,Imtx,Rety,Imty,U,tmax,wmin,wmax),'r')
	
	reswq  = np.asarray(G[Gstr]['ωqRe'])+1.j*np.asarray(G[Gstr]['ωqIm'])
	
	if spec == 'A1P':
		
		G = h5py.File(filename_wq(set,'IPE',L,tfc,tcc,tff,Retx,Imtx,Rety,Imty,U,tmax,wmin,wmax),'r')
		
		reswq += np.asarray(G[Gstr]['ωqRe'])+1.j*np.asarray(G[Gstr]['ωqIm'])
	
	return reswq

def axis_shenanigans(ax, XLABEL=True, YLABEL=True):
	
	ylim0, ylim1 = calc_ylim(U,spec)
	ax.set_ylim(ylim0,ylim1)
	ax.set_xlim(0,2*pi)
	if XLABEL:
		ax.set_xlabel('$k$')
	if YLABEL:
		ax.set_ylabel('$\omega$')
	ax.set_xticks(qticks) # labels=qlabels
	ax.set_xticklabels(qlabels)
	ax.grid(alpha=0.5)

if args.plot == 'cell':
	
	datawq = -1./pi* imag(open_Gwq(spec,0)) -1./pi* imag(open_Gwq(spec,1))
	
	fig, ax = plt.subplots()
	
	im = ax.imshow(datawq, origin='lower', interpolation='none', cmap=cm.terrain, aspect='auto', extent=[0,2*pi,wmin,wmax])
	fig.colorbar(im)
	
	axis_shenanigans(ax)
	
	plotname = spec+'cell_T=0'+'_U='+str(U)+'_tmax='+str(tmax)

if args.save:
	
	figname = plotname+'.pdf'
	savefig(figname, bbox_inches='tight')
	os.system('pdfcrop '+figname+' '+figname)

plt.show()


