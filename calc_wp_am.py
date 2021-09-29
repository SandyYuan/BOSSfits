import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import rc, rcParams
rcParams.update({'font.size': 12})

import numpy as np
import os
import sys
import time
import yaml
import h5py
import argparse
from pathlib import Path

from Corrfunc.theory import wp, xi


rpbins = np.logspace(-1, np.log10(30), 21)
# summit auto corr
pos_summit = np.load("../summit/data/data_gal_am_ph000_cleaned.npz")['x'] % 2000
wp_results = wp(2000, 30, 1, rpbins, pos_summit[:, 0], pos_summit[:, 1], pos_summit[:, 2], 
    verbose=False, output_rpavg=False) # this is all done in mpc / h
wp_summit = np.array([row[3] for row in wp_results])

# cosmos auto corr
wps_cosmos = 0
Ncosmos = 10
for whichsim in range(Ncosmos):
    pos_gals = np.load("../s3PCF_fenv/data/data_gal_am_"+str(whichsim)+".npz")['x']
    wp_results = wp(1100, 30, 1, rpbins, pos_gals[:, 0], pos_gals[:, 1], pos_gals[:, 2], 
        verbose=False, output_rpavg=False) # this is all done in mpc / h
    wps_cosmos += np.array([row[3] for row in wp_results])
wp_cosmos = wps_cosmos / Ncosmos

rmids = 0.5*(rpbins[1:] + rpbins[:-1])
fig = pl.figure(figsize = (4.3, 4))
pl.xlabel('$r_p$ ($h^{-1}$Mpc)')
pl.ylabel('$r_p w_p$ ($h^{-2}$Mpc$^2$)')
pl.plot(rmids, rmids * wp_summit, label = 'summit')
pl.plot(rmids, rmids * wp_cosmos, label = 'cosmos')
pl.xscale('log')
pl.legend(loc = 1, fontsize = 12)
pl.tight_layout()
fig.savefig("./plots/plot_autocorr_am.pdf", dpi = 200)


# cosmos particle loader
# get the particle positions in the sim
def part_pos(whichsim, params):
    # file directory
    mydir = '/mnt/gosling1/bigsim_products/AbacusCosmos_1100box_planck_products/'
    mysim = 'AbacusCosmos_1100box_planck_00'

    starttime = time.time()

    # we need to modify sim_name for halotools
    if sim_name.endswith("_00"):
        sim_name = sim_name[:-3]

    # first load a big catalog of halos and particles
    cats = Halotools.make_catalogs(sim_name=sim_name, phases = whichsim, 
                                cosmologies=0, redshifts=params['z'],
                                products_dir=products_dir,
                                halo_type='FoF', load_ptcl_catalog=True)

    # pull out the particle subsample 
    part_table = cats.ptcl_table


    return part_table