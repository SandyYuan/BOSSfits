#! /usr/bin/env python

import os
import time
import sys

import numpy as np
import argparse
import yaml
import dill
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib import rc, rcParams
rcParams.update({'font.size': 10})

from stochopy import MonteCarlo, Evolutionary

from stochopy import Evolutionary

from likelihood_boss import xirppi_Data, sigma_Data
from abacusnbody.hod.abacus_hod import AbacusHOD
from calc_sigma_fast import calc_Sigma

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/boss_lrg_xi.yaml'

def chi2(p, param_mapping, param_tracer, Data, Data_sigma, Ball, Sigma):
    print("evaulating ", p)
    # read the parameters 
    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        tracer_type = param_tracer[key]
        #tracer_type = param_tracer[params[mapping_idx, -1]]
        if key == 'sigma':
            Ball.tracers[tracer_type][key] = 10**p[mapping_idx]
        else:
            Ball.tracers[tracer_type][key] = p[mapping_idx]
                            
    # we need to determine the expected number density 
    Ball.tracers['LRG']['ic'] = 1
    ngal_dict = Ball.compute_ngal(Nthread = 32)[0]
    # we are only dealing with lrgs here
    N_lrg = ngal_dict['LRG']
    print("Nlrg ", N_lrg, "data density ", Data.num_dens_mean['LRG'])
    # print(Ball.tracers['LRG']['ic'], N_lrg, Data.num_dens_mean['LRG']*Ball.params['Lbox']**3)
    Ball.tracers['LRG']['ic'] = \
        min(1, Data.num_dens_mean['LRG']*Ball.params['Lbox']**3/N_lrg)

    theory_density = {
    'LRG': N_lrg * Ball.tracers['LRG']['ic']/Ball.params['Lbox']**3
    }

    # pass them to the mock dictionary
    mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread = 64)
    clustering = Ball.compute_xirppi(mock_dict, Ball.rpbins, Ball.pimax, Ball.pi_bin_size, Nthread = 16)
    lnP = Data.compute_likelihood(clustering, theory_density)

    # sigma
    mock_sigma = Sigma.run_hod(tracers = Ball.tracers, Nthread = 64)
    lnP_sigma = Data_sigma.compute_likelihood(mock_sigma, theory_density)

    return -2*(lnP + lnP_sigma)


def main(path2config):

    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    optimize_config_params = config['optimize_config_params']
    fit_params = config['optimize_params']    
    sigma_params = config['sigma_params'] 

    # create a new abacushod object and load the subsamples
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    newSigma = calc_Sigma(newBall, sigma_params)

    # read data parameters
    newData = xirppi_Data(data_params, HOD_params)
    newData_sigma = sigma_Data(data_params, HOD_params)

    # parameters to fit
    nparams = len(fit_params.keys())
    param_mapping = {}
    param_tracer = {}
    params = np.zeros((nparams, 2))
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        tracer_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_tracer[key] = tracer_type
        params[mapping_idx, :] = fit_params[key][1:-1]

    # Make path to output
    if not os.path.isdir(os.path.expanduser(optimize_config_params['path2output'])):
        try:
            os.makedirs(os.path.expanduser(optimize_config_params['path2output']))
        except:
            pass
        
    # where to record
    prefix_chain = os.path.join(os.path.expanduser(optimize_config_params['path2output']),
                                optimize_config_params['chainsPrefix'])

    popsize = optimize_config_params['popsize']
    max_iter = optimize_config_params['max_iter']
    ftol = optimize_config_params['ftol']

    ea = Evolutionary(chi2, args = [param_mapping, param_tracer, newData, newData_sigma, newBall, newSigma], 
        lower = params[:, 0], upper = params[:, 1], popsize = popsize, max_iter = max_iter)
    xopt, gfit = ea.optimize(solver = optimize_config_params['solver'])
    print(xopt, gfit)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    main(**args)
