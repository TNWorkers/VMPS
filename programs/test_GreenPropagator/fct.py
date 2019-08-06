#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc
from matplotlib import rcParams
from numpy import amin, amax, savetxt, arange, argmax
from scipy.optimize import curve_fit
import os, sys
import glob
import argparse
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties

x = [4., 3., 2., 1.]
N = 4
k = arange(N)

def dct1(x):
	y = zeros(2*N)
	y[:N] = x
	Y = fft(y)[:N]
	Y *= 2 * exp(-1j*pi*k/(2*N))
	return Y.real

def dct0(x):
	y = zeros(N)
	for j in range(N):
		for ik in range(N):
			y[ik] += x[j] * 2. * cos(pi*(j+0.5)*k[ik]/N)
	return y

print(dct0(x))
print(dct1(x))
