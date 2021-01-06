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
parser.add_argument('-spec', action='store', default='SSF')
parser.add_argument('-Ncells', action='store', type=int, default=8)
parser.add_argument('-Lcell', action='store', type=int, default=2)
parser.add_argument('-tfc', action='store', type=int, default=1)
parser.add_argument('-tcc', action='store', type=int, default=1)
parser.add_argument('-tff', action='store', type=int, default=0)
parser.add_argument('-Retx', action='store', type=int, default=0)
parser.add_argument('-Imtx', action='store', type=int, default=0)
parser.add_argument('-Rety', action='store', type=int, default=0)
parser.add_argument('-Imty', action='store', type=int, default=0)
parser.add_argument('-Ef', action='store', type=int, default=-2)
parser.add_argument('-Ec', action='store', type=int, default=0)
parser.add_argument('-U', action='store', type=int, default=8)
parser.add_argument('-V', action='store', type=int, default=0)
parser.add_argument('-beta', action='store', type=int, default=5)
parser.add_argument('-Ly', action='store', type=int, default=1)
parser.add_argument('-tolDeltaS', action='store', type=float, default=0.01)
parser.add_argument('-dt', action='store', type=float, default=0.025)
parser.add_argument('-tmax', action='store', type=int, default=4)
parser.add_argument('-index', action='store', type=int, default=0)
args = parser.parse_args()

set = args.set
spec = args.spec
plot = args.plot
index = args.index

U = args.U
V = args.V
beta = args.beta
tfc = args.tfc
tcc = args.tcc
tff = args.tff
Retx = args.Retx
Imtx = args.Imtx
Rety = args.Rety
Imty = args.Imty
Ef = args.Ef
Ec = args.Ec

dt = args.dt
tolDeltaS = args.tolDeltaS
tmax = args.tmax

Lcell = args.Lcell
Ncells = args.Ncells

def filename(set,spec,Lcell,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,beta,tmax):
	res = set+'/'
	res += spec
	res += '_tfc='+str(tfc)
	res += '_tcc='+str(tcc)
	res += '_tff='+str(tff)
	res += '_tx='+str(Retx)+","+str(Imtx)
	res += '_ty='+str(Rety)+","+str(Imty)
	res += '_Efc='+str(Ef)+","+str(Ec)
	res += '_U='+str(U)
	res += '_V='+str(V)
	res += '_beta='+str(beta)
	res += '_L='+str(Lcell)+'x'+str(Ncells)
	res += '_dLphys='+str(2)
	res += '_tmax='+str(tmax)
	res += '_Op=S'
	res += '.h5'
	print(res)
	return res

def open_Gwq (spec,index):
	
	Gstr = 'i='+str(index)
	
	G = h5py.File(filename(set,spec,Lcell,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,beta,tmax),'r')
	res  = np.asarray(G[Gstr])
	
	return res

def axis_shenanigans(ax, XLABEL=True, YLABEL=True):
	
	if XLABEL:
		ax.set_xlabel('$x$')
	if YLABEL:
		ax.set_ylabel('$t$')
	ax.grid(alpha=0.5)

if args.plot == 'cell':
	
	data = open_Gwq(spec,index)
	
	fig, ax = plt.subplots()
	
	im = ax.imshow(data[:,0:-1:2], origin='lower', interpolation='none', cmap=cm.terrain, aspect='auto', extent=[0,2*pi,0,tmax])
	fig.colorbar(im)
	
	axis_shenanigans(ax)
	
	plotname = spec+'cell_T=0'+'_U='+str(U)+'_tmax='+str(tmax)+'_beta='+str(beta)

if args.save:
	
	figname = plotname
	savefig(figname+'.pdf', bbox_inches='tight')
	savefig(figname+'.png', bbox_inches='tight')
	os.system('pdfcrop '+figname+'.pdf'+' '+figname+'.pdf')

plt.show()


