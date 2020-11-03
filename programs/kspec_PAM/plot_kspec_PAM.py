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
parser.add_argument('-L', action='store', type=int, default=2)
parser.add_argument('-Ncells', action='store', type=int, default=16)
parser.add_argument('-Ly', action='store', type=int, default=1)
parser.add_argument('-tol', action='store', type=float, default=0.01)
parser.add_argument('-dt', action='store', type=float, default=0.1)
parser.add_argument('-tmax', action='store', type=int, default=4)
parser.add_argument('-INT', action='store', default='OOURA')
parser.add_argument('-index', action='store', type=int, default=2)

parser.add_argument('-U', action='store', type=int, default=4)
parser.add_argument('-V', action='store', type=int, default=0)

parser.add_argument('-Ef', action='store', type=float, default=-2)
parser.add_argument('-Ec', action='store', type=float, default=0)

parser.add_argument('-tfc', action='store', type=float, default=0.5)
parser.add_argument('-tcc', action='store', type=float, default=1)
parser.add_argument('-tff', action='store', type=float, default=0)
parser.add_argument('-Retx', action='store', type=float, default=0)
parser.add_argument('-Imtx', action='store', type=float, default=0.5)
parser.add_argument('-Rety', action='store', type=float, default=0)
parser.add_argument('-Imty', action='store', type=float, default=0)

args = parser.parse_args()

set = args.set
spec = args.spec
plot = args.plot
INT = args.INT
index = args.index

U = round(args.U)
V = round(args.V)
tfc = round(args.tfc)
tcc = round(args.tcc)
tff = round(args.tff)
Retx = round(args.Retx)
Imtx = round(args.Imtx)
Rety = round(args.Rety)
Imty = round(args.Imty)
Ec = round(args.Ec)
Ef = round(args.Ef)

dt = args.dt
tolDeltaS = args.tol
tmax = args.tmax

L = args.L
Ns = L/2;
Ncells = args.Ncells

long_spec = {'A1P':'one-particle', 'PES':'photoemission', 'IPE':'inv. photoemission', 'SSF':'spin', 'PSZ':'charge', 'CSF':'charge', 'HSF':'hybridization'}

#def calc_ylim(UA,spec):
#	ylim0 = {'A1P':-10, 'PES':-10, 'IPE':-10, 'HSF':-10, 'CSF':-10, 'PSZ':-10}
#	ylim1 = {'A1P':+10, 'PES':+10, 'IPE':+10, 'HSF':+10, 'CSF':+10, 'PSZ':+10}
#	return ylim0[spec], ylim1[spec]

wmin = -10
wmax = +10

qticks = [0, pi/2, pi, 3*pi/2, 2*pi]
qlabels = ['$0$', '$\\frac{\pi}{2}$', '$\pi$', '$\\frac{3\pi}{2}$', '$2\pi$']

def H00(k):
	return Ec-2.*tcc*cos(k)

def H11(k):
	return Ef-2.*tff*cos(k)

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

def analytical_2p():
	
	kaxis, disp1, disp2 = analytical_disp()
	
	kvals = []
	Evals = []
	
	for k1,epsk1 in zip(kaxis,disp1):
		for k2,epsk2 in zip(kaxis,disp2):
		
			ktot = (k1+k2)%(2*pi)
			epstot = abs(epsk1)+abs(epsk2)
			
			kvals.append(ktot)
			Evals.append(epstot)
	
	return kvals, Evals

def filename_wq(set,spec,L,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,tmax,wmin,wmax):
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
#	res += '_dt='+str(dt)
#	res += '_tolΔS='+str(tolDeltaS)
	res += '_L='+str(L)+'x'+str(Ncells)
#	res += '_dLphys='+str(2)
	res += '_tmax='+str(tmax)
	res += '_INT='+INT
	res += '_qmin=0_qmax=2'
	res += '_wmin='+str(wmin)+'_wmax='+str(wmax)+'_Ns='+str(Ns)
	res += '.h5'
	print(res)
	return res

def filename_tx(set,spec,L,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,tmax):
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
	res += '_L='+str(L)+'x'+str(Ncells)
	res += '_tmax='+str(tmax)
	res += '_INT='+INT
	res += '.h5'
	print(res)
	return res

def open_Gwq (spec,index):
	
	Gstr = 'G'+str(index)+str(index)
	
	if spec == 'A1P':
		G = h5py.File(filename_wq(set,'PES',L,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,tmax,wmin,wmax),'r')
	else:
		G = h5py.File(filename_wq(set,spec,L,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,tmax,wmin,wmax),'r')
	
	res  = np.asarray(G[Gstr]['ωqRe'])+1.j*np.asarray(G[Gstr]['ωqIm'])
	
	if spec == 'A1P':
		
		G = h5py.File(filename_wq(set,'IPE',L,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,tmax,wmin,wmax),'r')
		
		res += np.asarray(G[Gstr]['ωqRe'])+1.j*np.asarray(G[Gstr]['ωqIm'])
	
	return res

def open_QDOS (spec,index):
	
	Gstr = 'G'+str(index)
	
	G = h5py.File(filename_wq(set,'PES',L,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,tmax,wmin,wmax),'r')
	res  = np.asarray(G[Gstr]['QDOS'])
	
	G = h5py.File(filename_wq(set,'IPE',L,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,tmax,wmin,wmax),'r')
	res += np.asarray(G[Gstr]['QDOS'])
	
	return res

def open_Gtx (spec,index):
	
	Gstr = 'G'+str(index)+str(index)
	G = h5py.File(filename_tx(set,spec,L,Ncells,tfc,tcc,tff,Retx,Imtx,Rety,Imty,Ef,Ec,U,V,tmax),'r')
	res  = np.asarray(G[Gstr]['txRe'])+1.j*np.asarray(G[Gstr]['txIm'])
#	if spec == 'PSZ':
#		res *= 2.
	return res

def axis_shenanigans(ax, XLABEL=True, YLABEL=True):
	
	ylim0, ylim1 = wmin, wmax #calc_ylim(U,spec)
	ax.set_ylim(ylim0,ylim1)
	ax.set_xlim(0,2*pi)
	if XLABEL:
		ax.set_xlabel('$k$')
	if YLABEL:
		ax.set_ylabel('$\omega$')
	ax.set_xticks(qticks) # labels=qlabels
	ax.set_xticklabels(qlabels)
	ax.grid(alpha=0.5)

if args.plot == 'freq':
	
	if index>1:
		data = -1./pi* imag(open_Gwq(spec,0)) -1./pi* imag(open_Gwq(spec,1))
	else:
		data = -1./pi* imag(open_Gwq(spec,index))
	
	print("wpoints=",len(data[:,0]),"qpoints=",len(data[0,:]))
	
	fig, ax = plt.subplots()
	im = ax.imshow(data, origin='lower', interpolation='none', cmap=cm.terrain, aspect='auto', extent=[0,2*pi,wmin,wmax])
	fig.colorbar(im)
	axis_shenanigans(ax)
	
	plotname = spec+'cell_T=0'+'_U='+str(U)+'_tmax='+str(tmax)
	
	if spec == 'A1P' or spec == 'PES' or spec == 'IPE':
		kaxis, disp1, disp2 = analytical_disp()
		ax.plot(kaxis, disp1, c='r')
		ax.plot(kaxis, disp2, c='r')
	else:
		kvals, Evals = analytical_2p()
#		ax.scatter(kvals, np.asarray(Evals), c='r', s=0.1)

elif args.plot == 'QDOS':
	
	data = open_QDOS('A1P',0) + open_QDOS('A1P',1)
	
	waxis = linspace(wmin,wmax,len(data),endpoint=True)
	fig, ax = plt.subplots()
	plt.plot(waxis, data, marker='.')
	plt.grid()

elif args.plot == 'time':
	
	if index>1:
		data = open_Gtx(spec,0) + open_Gtx(spec,1)
	else:
		data = open_Gtx(spec,index)
	
	fig, axs = plt.subplots(1, 2)
	im = axs[0].imshow(real(data), vmin=-0.05, vmax=0.05, origin='lower', interpolation='none', cmap=cm.gnuplot, aspect='auto', extent=[0,2*pi,0,tmax])
	im = axs[1].imshow(imag(data), vmin=-0.05, vmax=0.05, origin='lower', interpolation='none', cmap=cm.gnuplot, aspect='auto', extent=[0,2*pi,0,tmax])
	fig.colorbar(im)
	plt.title(spec)
#	axis_shenanigans(ax)

elif args.plot == 'tslice':
	
	if index>1:
		data = open_Gtx(spec,0) + open_Gtx(spec,1)
	else:
		data = open_Gtx(spec,index)
	
	fig, ax = plt.subplots()
	plt.plot(real(data[4,:]), marker='.')
	plt.plot(imag(data[4,:]), marker='.')
	plt.title(spec)
#	axis_shenanigans(ax)

if args.save:
	
	figname = plotname+'.pdf'
	savefig(figname, bbox_inches='tight')
	os.system('pdfcrop '+figname+' '+figname)

plt.show()


