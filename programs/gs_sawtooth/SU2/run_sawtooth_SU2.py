#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

def round(x):
	if x==int(x):
		return int(x)
	else:
		return x

parser = argparse.ArgumentParser()
parser.add_argument('-save', action='store_true', default=False)
parser.add_argument('-L', action='store', type=int, default=100)
parser.add_argument('-Smin', action='store', type=int)
parser.add_argument('-Smax', action='store', type=int)
parser.add_argument('-THREADS', action='store', type=int, default=2)
parser.add_argument('-Mlimit', action='store', type=int, default=1000)
parser.add_argument('-J', action='store', type=float, default=-1)
parser.add_argument('-JpA', action='store', type=float, default=1)
parser.add_argument('-PBC', action='store', type=float, default=1)
parser.add_argument('-tol_eigval', action='store', type=float, default=1e-9)
parser.add_argument('-tol_state', action='store', type=float, default=1e-8)
parser.add_argument('-LOAD', action='store', type=bool, default=False)
parser.add_argument('-Mold', action='store', type=int, default=0)
parser.add_argument('-min_halfsweeps', action='store', type=int, default=36)
parser.add_argument('-max_halfsweeps', action='store', type=int, default=40)
parser.add_argument('-eps_truncWeight', action='store', type=int, default=0)
parser.add_argument('-Qinit', action='store', type=int, default=10)
parser.add_argument('-Minit', action='store', type=int, default=10)
parser.add_argument('-Mincr_abs', action='store', type=int, default=60)
parser.add_argument('-end_2site', action='store', type=int, default=0)
args = parser.parse_args()

wd = 'Mlimit='+str(args.Mlimit)

if not os.path.isdir(wd):
	os.mkdir(wd)
	os.mkdir(wd+'/obs')
	os.mkdir(wd+'/log')
	os.mkdir(wd+'/state')

tol_eigval = args.tol_eigval
tol_state = args.tol_state
min_halfsweeps = args.min_halfsweeps
max_halfsweeps = args.max_halfsweeps
Minit = args.Minit
Qinit = args.Qinit
Mincr_abs = args.Mincr_abs
eps_truncWeight = args.eps_truncWeight
end_2site = args.end_2site

for S in range(args.Smin,args.Smax+1):
	call = 'OMP_NUM_THREADS='+str(args.THREADS)+' ./SU2.out -JA='+str(args.J)+' -JB='+str(args.J)+' -JpA='+str(args.JpA)+' -JpB=0 -PBC='+str(args.PBC)+' -CALC_VAR=1 -L '+str(args.L)+' -S '+str(S)+' -Mlimit '+str(args.Mlimit)+' -Mincr_abs='+str(Mincr_abs)+' -wd '+wd+' -tol_eigval '+str(tol_eigval)+' -tol_state '+str(tol_state)+' -min_halfsweeps='+str(min_halfsweeps)+' -max_halfsweeps='+str(max_halfsweeps)+' -Qinit='+str(Qinit)+' -Minit='+str(Minit)+' -eps_truncWeight='+str(eps_truncWeight)+' -end_2site='+str(end_2site)
	
	if args.LOAD:
		call = call + ' -LOAD='+'./Mlimit='+str(args.Mold)+'/state/'+'gs_L='+str(args.L)+'_J='+str(round(args.J))+','+str(round(args.J))+'_Jprime='+str(round(args.JpA))+',0'+'_D=2_S='+str(S)+'_PBC=1_Mmax='+str(args.Mold)
		call = call + ' -max_halfsweeps=20'
	else:
		if args.Mlimit == 500:
			call = call + ' -max_halfsweeps=33'
	print(call)
	os.system(call)
