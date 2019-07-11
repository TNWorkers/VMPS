#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc
from matplotlib import rcParams
from numpy import amin, amax, savetxt
from scipy.optimize import curve_fit
import os, sys
import glob
import argparse
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties

#sys.path.insert(0, '../PYSNIP')
#import StringStuff

rc('text',usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('font',size=14)

dE_list = [0.5, 0.4, 0.3, 0.25, 0.2]
dE_listring = ','.join([str(dE) for dE in dE_list])

parser = argparse.ArgumentParser()
parser.add_argument('-save', action='store_true', default=False)
parser.add_argument('-Nt', action='store', default=100)
parser.add_argument('-tmax', action='store', default=10)
args = parser.parse_args()

filename = 'GwqIm_tmax='+str(args.tmax)+'_Nt='+str(args.Nt)+'_wmin=0_wmax=3_Nw=1000.dat'
data = loadtxt(filename)
#print(len(data[:,0]),len(data[0,:]))

im = imshow(-1./pi*data, origin='lower', interpolation='none', cmap=cm.inferno, aspect='auto', extent=[0,2*pi,0,3])
colorbar()

if args.save:
	
	print filename(-1,'.pdf')
	savefig(filename(-1,'.pdf'), bbox_inches='tight')
	os.system('pdfcrop '+filename(-1,'.pdf')+' '+filename(-1,'.pdf'))

plt.show()
