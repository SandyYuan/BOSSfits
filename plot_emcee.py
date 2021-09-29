import numpy as np
import os,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib import rc, rcParams
rcParams.update({'font.size': 20})

logls = np.loadtxt("./chains/test/wp_testprob.txt")
fig = pl.figure(figsize = (10, 3))
pl.plot(-logls[1000:], alpha = 0.5)
pl.yscale('log')
pl.tight_layout()
fig.savefig("./plots/plot_emcee_logl_wp.pdf", dpi = 200)