#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function #python2 is then compatible with python3 print syntax. The other way around is not possible.

import os, sys
import numpy as np
#from termcolor import colored, cprint
import matplotlib.pyplot as plt
from math import *
from cmath import *
from matplotlib import rc, rcParams
import h5py

sys.path.insert(0, '../PYSNIP')
#import folders
# import StringStuff

def myRound(x):
    if x==0.:
        return 0
    elif x==1.:
        return 1
    elif x==2.:
        return 2
    elif x==3.:
        return 3
    elif x==4.:
        return 4
    elif x==int(x):
        return int(x)
    else:
        return x

rc('text', usetex=True)
rc('font', **{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('font', size=12)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-U', action='store', default=2)
parser.add_argument('-V', action='store', default=2)
parser.add_argument('-J', action='store', default=0)
parser.add_argument('-set', action='store', default='SiSj')
parser.add_argument('-save', action='store', default=False)
args = parser.parse_args()

U = myRound(float(args.U))
V = myRound(float(args.V))
J = myRound(float(args.J))

f = h5py.File('./toydata/obs/U='+str(U)+'_V='+str(V)+'_J='+str(J)+'.h5','r')

Chis = []
logChis = []
S0 = []
S1 = []

for Chi in np.array(list(map(int,list(f)))):
	Chis.append(int(Chi))
Chis.sort()

for Chi in Chis:
	S0.append(f[str(Chi)]['Entropy'][0][0])
	S1.append(f[str(Chi)]['Entropy'][1][0])
	logChis.append(log(Chi))

fit0 = np.poly1d(np.polyfit(logChis, S0, 1))
fit1 = np.poly1d(np.polyfit(logChis, S1, 1))
print('fit S=c/3*log(χ)+γ:')
print('c=',np.real(fit0(0))/3, 'γ=',np.real(fit0(1)))
print('c=',np.real(fit1(0))/3, 'γ=',np.real(fit1(1)))

#plt.xlabel('$\ln \chi$')
#plt.ylabel('$S$')
#plt.plot(logChis, S0, marker='.', label='S0')
#plt.plot(logChis, S1, marker='.', label='S1')
#plt.plot(logChis, fit0(logChis), marker='.', label='fit0')
#plt.plot(logChis, fit1(logChis), marker='.', label='fit1')
#plt.legend()

print('χs=',Chis)
Chi = str(max(Chis))
print('using χ=', Chi)

print('Dmax=', f[Chi]['Dmax'][0])
print('Mmax=', f[Chi]['Mmax'][0])
print('err_eigval=', f[Chi]['err_eigval'][0])
print('err_state=', f[Chi]['err_state'][0])
print('err_var=', f[Chi]['err_var'][0])
print('S=', f[Chi]['Entropy'][0], f[Chi]['Entropy'][1])
print('nh=', f[Chi]['nh'][0], f[Chi]['nh'][1])
print('ns=', f[Chi]['ns'][0], f[Chi]['ns'][1])

datasets = ['SiSj', 'TiTj']
labels = {'SiSj':'$S(k)$', 'TiTj':'$T(k)$'}

def k (n,N):
	#return 2.*pi/N*(n-N/2) # from -pi to pi
	return 2.*pi/N*n # from 0 to 2*pi

for dataset in datasets:
	
	data = np.array(f[Chi][dataset])
	
	N = min(200,len(data[:,0]))
	
	setk = [0.j] * (N+1)
	kvals = [0] * (N+1)
	for n in range(N+1):
		kvals[n] = k(n,N)
	
	# to test the fft:
#	for n in range(N+1):
#		for i in range(N):
#			for j in range(N):
#				setk[n] = setk[n] + data[i,j] * exp(-1.j*kvals[n]*(i-j)) / N
#	setk = np.real(setk)
	
	fftres = np.fft.fft2(data) / N
	
	for n in range(N+1):
		setk[n] = np.real(fftres[n%N,(N-n)%N]) # fft uses exp(-i*k1*Ri)*exp(-i*k2*Ri), need k1=k, k2=-k
	
	plt.plot(kvals, setk, ls='-', marker='.', label=labels[dataset])
	print(dataset, 'max at k/π=', k(np.argmax(setk),N)/pi)
	
#	plt.ylabel('', fontsize=14)
	plt.xlabel('$k$', fontsize=14)
	plt.xticks([0, pi/2, pi, 3*pi/2, 2*pi], [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

plt.xlim(k(0,N), k(N,N))
plt.ylim(0)
plt.grid()
plt.legend()

if args.save:
	
	outfile = 'outfile'
	savefig(outfile+'.pdf', bbox_inches='tight')
	system('pdfcrop '+outfile+'.pdf '+outfile+'.pdf')

plt.show()

