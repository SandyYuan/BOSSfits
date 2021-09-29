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
from matplotlib import rc, rcParams
rcParams.update({'font.size': 10})

from abacusnbody.hod.abacus_hod import AbacusHOD

DEFAULTS = {}
DEFAULTS['path2config'] = '../abacusutils/scripts/hod/config/abacus_hod.yaml'

def subsample_mock(mock_dict, fraction, rseed):
    np.random.seed(rseed)

    new_dict = {}
    for etracer in mock_dict.keys():
        num_tracer = len(mock_dict[etracer]['x'])
        sub_indices = np.random.choice(num_tracer, int(fraction * num_tracer))

        newtracer_dict = {}
        for tracer_prop in ['x', 'y', 'z']:
            newtracer_dict[tracer_prop] = mock_dict[etracer][tracer_prop][sub_indices]
        new_dict[etracer] = newtracer_dict
    return new_dict

def main(path2config):

    # the boss cov diagonal 
    xi_sigmas_boss = np.sqrt(np.load("../s3PCF_fenv/data/data_xi_cov400_norm.npz")['diag'])

    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    
    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1)
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']
    
    # create a new abacushod object
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

    # determine the data / mock density ratio
    ngal_dict = newBall.compute_ngal(Nthread = 32)[0]
    # we are only dealing with lrgs here
    N_lrg = ngal_dict['LRG']
    Nratio = 3.0103e-4 * 2000**3 / N_lrg
    
    # throw away run for jit to compile, write to disk
    mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk = False, Nthread = 32)
    # mock_dict = newBall.gal_reader()
    start = time.time()
    xirppi = newBall.compute_xirppi(mock_dict, rpbins, pimax, pi_bin_size, Nthread = 32)['LRG_LRG']
    print("full sample xi, total time ", time.time() - start)

    test_fractions = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    seeds = np.array([599, 244, 872, 885, 419, 627, 911, 656,  19, 983, 743, 663, 743,
        48, 990, 250, 908, 150, 182, 501, 220, 628,  68, 331, 716, 760,
       209, 681, 504, 279, 880, 511, 779, 371, 957,  82, 827, 498, 402,
       142, 772, 589, 200, 423, 969, 839, 433, 567, 409, 109])
    mean_timings = np.zeros(len(test_fractions))
    xi_std = np.zeros((len(test_fractions), len(xirppi.flatten())))
    # test all the fractions
    for i, efraction in enumerate(test_fractions):

        # run the fit 10 times for timing
        timings = np.zeros(len(seeds))
        xis = np.zeros((len(seeds), len(xirppi.flatten())))
        for j in range(len(seeds)):
            sub_mock_dict = subsample_mock(mock_dict, efraction, seeds[j])

            start = time.time()
            xirppi = newBall.compute_xirppi(sub_mock_dict, rpbins, pimax, pi_bin_size, Nthread = 32)['LRG_LRG']
            delta_t = time.time() - start

            timings[j] = delta_t
            xis[j] = xirppi.flatten()
            print("Done xi, total time ", delta_t)

        print("fraction ", efraction, "mean timing ", np.mean(timings), "std ", np.std(timings))
        mean_timings[i] = np.mean(timings)
        xi_std[i] = np.std(xis, axis = 0)

    fig = pl.figure(figsize = (12, 4))
    pl.xlabel('bins')
    pl.ylabel('shot noise / boss variance')
    pl.axhline(y = 1, xmin = 0, xmax = 1, ls = '--', alpha = 0.5)
    pl.axhline(y = 0.5, xmin = 0, xmax = 1, ls = '--', alpha = 0.5)
    for i, efraction in enumerate(test_fractions):
        pl.plot(xi_std[i] / xi_sigmas_boss, label = '$N/N_{\\mathrm{data}}$ = '+str(efraction/Nratio)[:5], alpha = 0.7)
    pl.legend(loc = 'best')
    pl.tight_layout()
    fig.savefig('./plots/plot_test_subsampling.pdf', dpi = 200)

    fig, ax1 = pl.subplots(figsize=(4.3, 3.4))

    color = 'tab:red'
    ax1.plot(test_fractions/Nratio, mean_timings, color = color, marker = 'o')
    ax1.set_xlabel("$N/N_{\\mathrm{data}}$")
    ax1.set_ylabel("timing", color = color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(test_fractions/Nratio, np.mean(xi_std / xi_sigmas_boss, axis = 1), color = color, marker = 's')
    ax2.set_ylabel("error ratio", color = color)
    ax2.tick_params(axis='y', labelcolor=color)

    pl.tight_layout()
    fig.savefig("./plots/plot_fraction_timing.pdf", dpi = 200)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":


    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())
    main(**args)
