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

parser = argparse.ArgumentParser()
parser.add_argument('-save', action='store_true', default=False)
args = parser.parse_args()

Ln = [12, 20, 24, 36, 48, 64]
xn = 1./np.asarray(Ln)
ngap = [0.04019733811414383, 0.03483551750387903, 0.03327098907131187, 0.02924924884076319, 0.02571139782409659, 0.02162907650897239]

Lt = [12, 20, 24, 36, 48, 64, 128]
xt = 1./np.asarray(Lt)
tgap = [0.01262991644805211, 0.007718958616976579, 0.006552585445227521, 0.00519644680080944, 0.004946076259798815, 0.004866073764517864, 0.004208506353052144]

def func(x, a, b, c):
	return a*x*x + b*x + c

params_n, cov_n = curve_fit(func, xn, ngap)
print(params_n)
print('interpolated neutral gap=',params_n[-1])

params_t, cov_t = curve_fit(func, xt, tgap)
print(params_t)
print('interpolated triplet gap=',params_t[-1])

plt.plot(xn, ngap, marker='o', ls='', label='neutral')
plt.plot(xt, tgap, marker='o', ls='', label='triplet')

xn = np.append(xn,0)
xt = np.append(xt,0)

plt.plot(xn, func(xn,*params_n), ls='--')
plt.plot(xt, func(xt,*params_t), ls='--')

plt.xlabel('$1/L$')
plt.ylabel('neutral gap')
plt.title('$U=4$, $E_f=-2$, $t_{fc}=0.5$')
plt.grid()
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.legend()

if args.save:
	
	filename = 'gap'
	savefig(filename+'.pdf', bbox_inches='tight')
	savefig(filename+'.png', bbox_inches='tight')
	os.system('pdfcrop '+filename+'.pdf'+' '+filename+'.pdf')

plt.show()
