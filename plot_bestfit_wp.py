#! /usr/bin/env python

import os
import time
import sys

import numpy as np
import argparse
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib import rc, rcParams
rcParams.update({'font.size': 10})

from likelihood_boss import xirppi_Data, wp_Data
from abacusnbody.hod.abacus_hod import AbacusHOD

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/boss_lrg_wp.yaml'

def inrange(p, params):
    return np.all((p<=params[:, 2]) & (p>=params[:, 1]))

def plot_wp(p, params, param_mapping, param_tracer, Data, Ball):
    print("plot wp")
    if inrange(p, params):
        # read the parameters 
        for key in param_mapping.keys():
            mapping_idx = param_mapping[key]
            tracer_type = param_tracer[key]
            #tracer_type = param_tracer[params[mapping_idx, -1]]
            Ball.tracers[tracer_type][key] = p[mapping_idx]
            
        # pass them to the mock dictionary
        print(Ball.tracers)

        Ball.tracers['LRG']['ic'] = 1
        # we need to determine the expected number density 
        ngal_dict = Ball.compute_ngal(Nthread = 32)[0]
        # we are only dealing with lrgs here
        N_lrg = ngal_dict['LRG']
        print("Nlrg ", N_lrg)
        # print(Ball.tracers['LRG']['ic'], N_lrg, Data.num_dens_mean['LRG']*Ball.params['Lbox']**3)
        Ball.tracers['LRG']['ic'] = \
            min(1, Data.num_dens_mean['LRG']*Ball.params['Lbox']**3/N_lrg)
        theory_density = {
        'LRG': N_lrg * Ball.tracers['LRG']['ic']/Ball.params['Lbox']**3
        }

        mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread = 64)
        clustering = Ball.compute_wp(mock_dict, Ball.rpbins, Ball.pimax, Ball.pi_bin_size, Nthread = 16)

        Data.compute_likelihood(clustering, theory_density)

        rs = 0.5*(Ball.rpbins[1:] + Ball.rpbins[:-1])
        # plotting utility
        for etracer in clustering.keys():
            lrgcov = np.linalg.inv(Data.icov[etracer])
            print(clustering[etracer], Data.clustering[etracer])
            delta = (rs*clustering[etracer] - rs*Data.clustering[etracer])[2:]
            print(np.einsum('i,ij,j', delta, Data.icov[etracer][2:, 2:], delta))

            fig = pl.figure(figsize = (5, 4))
            pl.plot(rs, rs*clustering[etracer], label = "cleaned", alpha = 0.8)
            # uncleaned = np.load("./data_wp_lrg_noclean.npz")
            # pl.plot(uncleaned['rs'], uncleaned['rwp'], label = "uncleaned", alpha = 0.8)
            pl.errorbar(rs, rs*Data.clustering[etracer], yerr = np.sqrt(np.diagonal(lrgcov)), label = 'CMASS', alpha = 0.8)
            pl.xscale('log')
            pl.xlabel('$r$ [$h^{-1}$Mpc]')
            pl.ylabel('$r w_p$ [($h^{-1}$Mpc)$^2$]')
            pl.legend(loc='best')
            pl.tight_layout()
            fig.savefig("./plots/plot_bestfit_wp_"+etracer+"_uncleaned.pdf", dpi = 200)
            # np.savez("./data_wp_lrg_noclean", rs = rs, rwp = rs*clustering[etracer])


def main(path2config):

    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    ch_config_params = config['ch_config_params']
    fit_params = config['fit_params']    

    # where to record
    prefix_chain = os.path.join(os.path.expanduser(ch_config_params['path2output']),
                                ch_config_params['chainsPrefix'])
    my_chain = np.loadtxt(prefix_chain+'.txt')
    my_logLs = np.loadtxt(prefix_chain+'prob.txt')
    params_best = my_chain[np.argmax(my_logLs)]
    print(params_best, my_logLs[np.argmax(my_logLs)])


    # create a new abacushod object and load the subsamples
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    newBall.tracers['LRG']['ic'] = 1
    # we need to determine the expected number density 
    ngal_dict = newBall.compute_ngal(Nthread = 32)[0]
    # we are only dealing with lrgs here
    N_lrg = ngal_dict['LRG']
    print("Nlrg ", N_lrg)
    
    # read data parameters
    newData = wp_Data(data_params, HOD_params)

    # parameters to fit
    nparams = len(fit_params.keys())
    param_mapping = {}
    param_tracer = {}
    params = np.zeros((nparams, 4))
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        tracer_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_tracer[key] = tracer_type
        params[mapping_idx, :] = fit_params[key][1:-1]


    plot_wp(params_best, params, param_mapping, param_tracer, newData, newBall)


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    main(**args)

