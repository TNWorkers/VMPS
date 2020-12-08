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
parser.add_argument('-Ncells', action='store', type=int, default=8)
parser.add_argument('-Lcell', action='store', type=int, default=2)
parser.add_argument('-tfc', action='store', type=int, default=1)
parser.add_argument('-tcc', action='store', type=int, default=0)
parser.add_argument('-tff', action='store', type=int, default=0)
parser.add_argument('-Retx', action='store', type=int, default=1)
parser.add_argument('-Imtx', action='store', type=int, default=0)
parser.add_argument('-Rety', action='store', type=int, default=0)
parser.add_argument('-Imty', action='store', type=int, default=0)
parser.add_argument('-Ef', action='store', type=int, default=0)
parser.add_argument('-Ec', action='store', type=int, default=0)
parser.add_argument('-U', action='store', type=int, default=8)
parser.add_argument('-V', action='store', type=int, default=0)
parser.add_argument('-beta', action='store', type=float, default=1)
parser.add_argument('-Ly', action='store', type=int, default=1)
parser.add_argument('-tolDeltaS', action='store', type=float, default=0.01)
parser.add_argument('-dt', action='store', type=float, default=0.025)
parser.add_argument('-tmax', action='store', type=int, default=8)
parser.add_argument('-INT', action='store', default='OOURA')
parser.add_argument('-index', action='store', type=int, default=0)
args = parser.parse_args()

set = args.set
spec = args.spec
plot = args.plot
INT = args.INT
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

long_spec = {'A1P':'one-particle', 'PES':'photoemission', 'IPE':'inv. photoemission', 'SSF':'spin', 'PSZ':'charge'}

def calc_ylim(spec):
#	ylim0 = {'A1P':-10, 'PES':-10, 'IPE':-10}
#	ylim1 = {'A1P':+10, 'PES':+10, 'IPE':+10}
#	return ylim0[spec], ylim1[spec]
	return -10, +10

wmin = -10
wmax = +10

qticks = [0, pi/2, pi, 3*pi/2, 2*pi]
qlabels = ['$0$', '$\\frac{\pi}{2}$', '$\pi$', '$\\frac{3\pi}{2}$', '$2\pi$']

def H00(k):
	return -2.*tcc*cos(k)

def H11(k):
	return -2.*tff*cos(k)

def H01(k):
	return -tfc-(Retx+1.j*Imtx)*exp(+1.j*k)-(Rety+1.j*Imty)*exp(-1.j*k)

tau0 = np.matrix([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]])
tau1 = np.matrix([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]])
tau2 = np.matrix([[0.+0.j, 0.-1.j], [0.+1.j, 0.+0.j]])
tau3 = np.matrix([[1.+0.j, 0.+0.j], [0.+0.j, -1.+0.j]])

def analytical_disp():
	
	kaxis = linspace(0, 2*pi, 101, endpoint=True)
	disp1 = np.zeros((101,1))
	disp2 = np.zeros((101,1))
	
	for ik,k in enumerate(kaxis):
		
		Heff = np.matrix([[H00(k), H01(k)], [conj(H01(k)), H11(k)]])
		
		c0 = 0.5*np.trace(tau0*Heff)
		c1 = 0.5*np.trace(tau1*Heff)
		c2 = 0.5*np.trace(tau2*Heff)
		c3 = 0.5*np.trace(tau3*Heff)
		b1 = real(c1)
		d1 = imag(c1)
		b2 = real(c2)
		d2 = imag(c2)
		b3 = real(c3)
		d3 = imag(c3)
		
		bnorm = b1*b1+b2*b2+b3*b3
		dnorm = d1*d1+d2*d2+d3*d3
		bddot = b1*d1+b2*d2+b3*d3
		
		disp1[ik] = real(c0 + sqrt(bnorm-dnorm+2.j*bddot))
		disp2[ik] = real(c0 - sqrt(bnorm-dnorm+2.j*bddot))
		
	return kaxis, disp1, disp2


def filename_wq(set,spec,Lcell,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,beta,tmax,wmin,wmax):
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
	res += '_β='+str(beta)
	res += '_L='+str(Lcell)+'x'+str(Ncells)
	res += '_dLphys='+str(2)
	res += '_tmax='+str(tmax)
	res += '_INT='+INT
	res += '_qmin=0_qmax=2'
	res += '_wmin='+str(wmin)+'_wmax='+str(wmax)
	res += '_Ns=1'
	res += '.h5'
	print(res)
	return res

def open_Gwq (spec,index):
	
	Gstr = 'G'+str(index)+str(index)
	
	if spec == 'A1P':
		G = h5py.File(filename_wq(set,'PES',Lcell,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,beta,tmax,wmin,wmax),'r')
	else:
		G = h5py.File(filename_wq(set,spec,Lcell,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,beta,tmax,wmin,wmax),'r')
	
	reswq  = np.asarray(G[Gstr]['ωqRe'])+1.j*np.asarray(G[Gstr]['ωqIm'])
	
	if spec == 'A1P':
		
		G = h5py.File(filename_wq(set,'IPE',Lcell,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,beta,tmax,wmin,wmax),'r')
		
		reswq += np.asarray(G[Gstr]['ωqRe'])+1.j*np.asarray(G[Gstr]['ωqIm'])
	
	return reswq

def axis_shenanigans(ax, XLABEL=True, YLABEL=True):
	
	ylim0, ylim1 = calc_ylim(spec)
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
	
	kaxis, disp1, disp2 = analytical_disp()
	ax.plot(kaxis, disp1, c='r')
	ax.plot(kaxis, disp2, c='r')

if args.save:
	
	figname = plotname+'.pdf'
	savefig(figname, bbox_inches='tight')
	os.system('pdfcrop '+figname+' '+figname)

plt.show()


