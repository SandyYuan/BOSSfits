#!/usr/bin/env python3
'''
This is a script for generating HOD mock catalogs.

Usage
-----
$ python ./run_hod.py --help
'''

import os
import glob
import time

import yaml
import numpy as np
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import rc, rcParams
rcParams.update({'font.size': 11})


from abacusnbody.hod.abacus_hod import AbacusHOD

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/boss_lrg_xi.yaml'

def main(path2config):

    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    
    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    write_to_disk = HOD_params['write_to_disk']
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1)
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']
    
    # create a new abacushod object
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    
    # throw away run for jit to compile, write to disk
    mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk = False, Nthread = 16)
    ngal_dict = newBall.compute_ngal()
    print(ngal_dict[0]['LRG']/2000**3)
    newBall.tracers['LRG']['ic'] = 3/12.4

    if os.path.exists("./data/plot_timing_thread.npz"):
        meantime_hod = np.load("./data/plot_timing_thread.npz")['y1']
        meantime_cf = np.load("./data/plot_timing_thread.npz")['y2']
        threads = np.load("./data/plot_timing_thread.npz")['x']
    else:
        # run the fit 10 times for timing
        threads = np.array([1, 2, 4, 8, 16, 32, 64])
        meantime_hod = np.zeros(7)
        meantime_cf = np.zeros(7)
        for whichtest, nthread in enumerate(threads):
            print(nthread)
            Ntest = 20
            hodtime = 0
            cftime = 0
            for i in range(Ntest):
                start = time.time()
                mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk, Nthread = nthread)
                newhodtime = time.time() - start
                hodtime += newhodtime
                print("Done hod, took time ", newhodtime)
                start = time.time()
                # ngal_dict = newBall.compute_ngal()
                # print("Done ngal, took time ", time.time() - start, ngal_dict)
                xirppi = newBall.compute_xirppi(mock_dict, rpbins, pimax, pi_bin_size, Nthread = nthread)
                deltat = time.time() - start
                cftime += deltat
                print("Done xi, total time ", deltat)
            meantime_hod[whichtest] = hodtime / Ntest
            meantime_cf[whichtest] = cftime / Ntest
            print("meantime ", hodtime / Ntest, cftime / Ntest)

        np.savez("./data/plot_timing_thread", x = threads, y1 = meantime_hod, y2 = meantime_cf)

    fig = pl.figure(figsize = (4, 4)) 
    pl.plot(threads, meantime_hod, alpha = 0.7, marker = 'o', label = 'HOD')
    pl.plot(threads, meantime_cf, alpha = 0.7, marker = 'o', label = 'Corrfunc')
    pl.plot(threads, meantime_hod+meantime_cf, alpha = 0.7, marker = 'o', label = 'Total')
    pl.axhline(y = np.min(meantime_hod+meantime_cf), xmin = 0, xmax = 1, ls = '--', alpha = 0.5)
    pl.xscale('log')
    pl.yscale('log')
    pl.ylim(0.1, 5)
    pl.legend(loc = 'best')
    pl.xlabel('$N_\mathrm{thread}$')
    pl.ylabel('time (s)')
    pl.title('$n_g = 3.0\\times 10^{-4}h^3$Mpc$^{-3}$')
    pl.tight_layout()
    fig.savefig("./plots/plot_timing_thread.pdf", dpi = 200) 

    # # timing vs density
    # densities = np.logspace(-5, np.log10(1.2e-3), num = 8)
    # meantime_hod = np.zeros(len(densities))
    # meantime_cf = np.zeros(len(densities))
    # for whichtest, edensity in enumerate(densities):
    #     print(edensity)
    #     newBall.tracers['LRG']['ic'] = edensity/1.24e-3
    #     Ntest = 20
    #     hodtime = 0
    #     cftime = 0
    #     for i in range(Ntest):
    #         start = time.time()
    #         mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk, Nthread = 32)
    #         newhodtime = time.time() - start
    #         hodtime += newhodtime
    #         print("Done hod, took time ", newhodtime)
    #         start = time.time()
    #         # ngal_dict = newBall.compute_ngal()
    #         # print("Done ngal, took time ", time.time() - start, ngal_dict)
    #         xirppi = newBall.compute_xirppi(mock_dict, rpbins, pimax, pi_bin_size, Nthread = 32)
    #         deltat = time.time() - start
    #         cftime += deltat
    #         print("Done xi, total time ", deltat)
    #     meantime_hod[whichtest] = hodtime / Ntest
    #     meantime_cf[whichtest] = cftime / Ntest
    #     print("meantime ", hodtime / Ntest, cftime / Ntest)
    # np.savez("./data/plot_timing_density", x = densities, y1 = meantime_hod, y2 = meantime_cf)

    # fig = pl.figure(figsize = (4, 4)) 
    # pl.plot(densities, meantime_hod, alpha = 0.7, marker = 'o', label = 'HOD')
    # pl.plot(densities, meantime_cf, alpha = 0.7, marker = 'o', label = 'Corrfunc')
    # pl.plot(densities, meantime_hod+meantime_cf, alpha = 0.7, marker = 'o', label = 'Total')
    # # pl.axhline(y = np.min(meantime_hod+meantime_cf), xmin = 0, xmax = 1, ls = '--', alpha = 0.5)
    # pl.axvline(x = 3e-4, ymin = 0, ymax = 1, ls = '--', alpha = 0.5)
    # pl.xscale('log')
    # pl.yscale('log')
    # pl.legend(loc = 'best')
    # pl.xlabel('$n_g$ ($h^3$Mpc$^{-3}$)')
    # pl.ylabel('time (s)')
    # pl.title('$N_\mathrm{thread} = 32$')
    # pl.tight_layout()
    # fig.savefig("./plots/plot_timing_density.pdf", dpi = 200) 

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":


    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())
    main(**args)
