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

def round(x):
	if x==int(x):
		return int(x)
	else:
		return x

rc('text',usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('font',size=10)
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

parser = argparse.ArgumentParser()
parser.add_argument('-save', action='store_true', default=False)
parser.add_argument('-set', action='store', default='.')
parser.add_argument('-plot', action='store', default='freq') # freq, time
parser.add_argument('-spec', action='store', default='A1P')
parser.add_argument('-L', action='store', type=int, default=12)
parser.add_argument('-tmax', action='store', type=int, default=4)
parser.add_argument('-INT', action='store', default='OOURA')
parser.add_argument('-j0', action='store', type=int, default=6)

parser.add_argument('-Ufile', action='store', default='U')
parser.add_argument('-tfile', action='store', default='t')
parser.add_argument('-Efile', action='store', default='E')

args = parser.parse_args()

set = args.set
spec = args.spec
plot = args.plot
INT = args.INT

tmax = args.tmax
L = args.L
Ufile = args.Ufile
tfile = args.tfile
Efile = args.Efile
j0 = args.j0

long_spec = {'A1P':'one-particle', 'PES':'photoemission', 'IPE':'inv. photoemission', 'SSF':'spin', 'PSZ':'charge', 'CSF':'charge', 'HSF':'hybridization'}

wmin = -10
wmax = +10

def filename_w (spec):
	res = set+'/'
	res += spec
	res += '_Ufile='+str(Ufile)
	res += '_tfile='+str(tfile)
	res += '_Efile='+str(Efile)
	res += '_L='+str(L)
	res += '_j0='+str(j0)
	res += '_tmax='+str(tmax)
	res += '_INT='+INT
	res += '_qmin=0_qmax=2'
	res += '_wmin='+str(wmin)+'_wmax='+str(wmax)
	res += '.h5'
	print(res)
	return res

def filename_t (spec):
	res = set+'/'
	res += spec
	res += '_Ufile='+str(Ufile)
	res += '_tfile='+str(tfile)
	res += '_Efile='+str(Efile)
	res += '_L='+str(L)
	res += '_j0='+str(j0)
	res += '_tmax='+str(tmax)
	res += '.h5'
	print(res)
	return res

def open_Gwq(spec):
	
	Gstr = 'G'
	
	if spec == 'A1P':
		G = h5py.File(filename_w('PES'),'r')
	else:
		G = h5py.File(filename_w(spec),'r')
	
	res  = np.asarray(G[Gstr]['ωxRe'])+1.j*np.asarray(G[Gstr]['ωxIm'])
	
	if spec == 'A1P':
		
		G = h5py.File(filename_w('IPE'),'r')
		
		res += np.asarray(G[Gstr]['ωxRe'])+1.j*np.asarray(G[Gstr]['ωxIm'])
	
	return res

def open_Gtx (spec):
	
	Gstr = 'G'
	G = h5py.File(filename_t(spec),'r')
	res  = np.asarray(G[Gstr]['txRe'])+1.j*np.asarray(G[Gstr]['txIm'])
	return res

def axis_shenanigans(ax, XLABEL=True, YLABEL=True):
	
	ylim0, ylim1 = wmin, wmax
	ax.set_ylim(ylim0,ylim1)
	ax.set_xlim(1,L)
	if XLABEL:
		ax.set_xlabel('$x$')
	if YLABEL:
		ax.set_ylabel('$\omega$')
	ax.grid(alpha=0.5)

if args.plot == 'freq':
	
	data = -1./pi* imag(open_Gwq(spec))
	
	fig, ax = plt.subplots()
	im = ax.imshow(data, origin='lower', interpolation='none', cmap=cm.terrain, aspect='auto', extent=[1,L,wmin,wmax])
	fig.colorbar(im)
	axis_shenanigans(ax)

elif args.plot == 'time':
	
	data = open_Gtx(spec)
	
	fig, axs = plt.subplots(1,2)
	im = axs[0].imshow(real(data), vmin=-0.05, vmax=0.05, origin='lower', interpolation='none', cmap=cm.gnuplot, aspect='auto', extent=[1,L,0,tmax])
	im = axs[1].imshow(imag(data), vmin=-0.05, vmax=0.05, origin='lower', interpolation='none', cmap=cm.gnuplot, aspect='auto', extent=[1,L,0,tmax])
	fig.colorbar(im)
	plt.title(spec)

if args.plot == 'j0freq':
	
	data = -1./pi* imag(open_Gwq(spec))
	
	waxis = linspace(wmin,wmax,len(data[:,0]),endpoint=True)
	
	fig, ax = plt.subplots()
	plt.plot(waxis, real(data[:,j0]))
	plt.plot(waxis, imag(data[:,j0]))
	plt.grid()
	plt.title(spec)

elif args.plot == 'j0t':
	
	data = open_Gtx(spec)
	
	fig, ax = plt.subplots()
	plt.plot(real(data[:,j0]), label='Re')
	plt.plot(imag(data[:,j0]), label='Im')
	plt.grid()
	plt.title(spec)

if args.save:
	
	figname = plotname
	savefig(figname+'.png', bbox_inches='tight')
	savefig(figname+'.pdf', bbox_inches='tight')
	os.system('pdfcrop '+figname+'.pdf'+' '+figname+'.pdf')

plt.show()


