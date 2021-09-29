import os
import glob
import time
import timeit
from pathlib import Path
import h5py
import yaml

import numpy as np
import h5py
import asdf
import argparse
from astropy.io import ascii

import numba
from numba import njit, jit

from wquantiles import quantile_1D

from abacusnbody.hod.abacus_hod import AbacusHOD
from likelihood_boss import sigma_Data
from calc_sigma_fast import calc_Sigma



def main(path2config, mock_key = 'LRG', combo_key = 'LRG_LRG'):
    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    dynesty_config_params = config['dynesty_config_params']
    fit_params = config['dynesty_fit_params']   
    sigma_params = config['sigma_params'] 

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

    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    newData = sigma_Data(data_params, HOD_params)
    newSigma = calc_Sigma(newBall, sigma_params)

    # load chains
    prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
                                dynesty_config_params['chainsPrefix'])

    datafile = np.load(prefix_chain+'_results.npz', allow_pickle = True)
    res1 = datafile['res'].item()

    # print the max logl fit
    logls = res1['logl']
    indmax = np.argmax(logls)
    hod_params = res1['samples'][indmax]
    print("max logl fit ", hod_params, logls[indmax])

    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        tracer_type = param_tracer[key]
        #tracer_type = param_tracer[params[mapping_idx, -1]]
        if key == 'sigma':
            newBall.tracers[tracer_type][key] = 10**hod_params[mapping_idx]
        else:
            newBall.tracers[tracer_type][key] = hod_params[mapping_idx] 
        print(key, newBall.tracers[tracer_type][key])

    # we need to determine the expected number density 
    newBall.tracers[mock_key]['ic'] = 1
    ngal_dict = newBall.compute_ngal(Nthread = 32)[0]
    # we are only dealing with lrgs here
    N_lrg = ngal_dict[mock_key]
    print("Nlrg ", N_lrg, "data density ", newData.num_dens_mean[mock_key])
    # print(Ball.tracers['LRG']['ic'], N_lrg, Data.num_dens_mean['LRG']*Ball.params['Lbox']**3)
    newBall.tracers[mock_key]['ic'] = \
        min(1, newData.num_dens_mean[mock_key]*newBall.params['Lbox']**3/N_lrg)

    newSigma = calc_Sigma(newBall, sigma_params)
    delsigma_best = newSigma.run_hod(Nthread = 64)[mock_key]

    delsigma_all = np.empty((len(res1['samples']), len(delsigma_best)))
    # now run all the points
    for i in range(len(res1['samples'])):
        ehod = res1['samples'][i]
        for key in param_mapping.keys():
            mapping_idx = param_mapping[key]
            tracer_type = param_tracer[key]
            #tracer_type = param_tracer[params[mapping_idx, -1]]
            if key == 'sigma':
                newBall.tracers[tracer_type][key] = 10**ehod[mapping_idx]
            else:
                newBall.tracers[tracer_type][key] = ehod[mapping_idx] 

        # we need to determine the expected number density 
        newBall.tracers[mock_key]['ic'] = 1
        ngal_dict = newBall.compute_ngal(Nthread = 32)[0]
        # we are only dealing with lrgs here
        N_lrg = ngal_dict[mock_key]
        print("Nlrg ", N_lrg, "data density ", newData.num_dens_mean[mock_key])
        # print(Ball.tracers['LRG']['ic'], N_lrg, Data.num_dens_mean['LRG']*Ball.params['Lbox']**3)
        newBall.tracers[mock_key]['ic'] = \
            min(1, newData.num_dens_mean[mock_key]*newBall.params['Lbox']**3/N_lrg)
        delsigma_all[i] = newSigma.run_hod(tracers = newBall.tracers, Nthread = 64)[mock_key]

    # what are the weights
    logwt = res1['logwt']

    # compute median and quantiles. 
    delsigma_median = np.zeros(len(delsigma_best))
    delsigma_lo = np.zeros(len(delsigma_best))
    delsigma_hi = np.zeros(len(delsigma_best))
    for j in range(len(delsigma_best)):
        delsigma_bin = delsigma_all[:, j]
        delsigma_median[j] = quantile_1D(delsigma_bin, np.exp(logwt), 0.5)
        delsigma_lo[j] = quantile_1D(delsigma_bin, np.exp(logwt), 0.16)
        delsigma_hi[j] = quantile_1D(delsigma_bin, np.exp(logwt), 0.84)

    np.savez("./data_lensing/data_"+dynesty_config_params['chainsPrefix']+"_werror", rs = newData.rs[combo_key], delta_sig = delsigma_median,
        delta_sig_lo = delsigma_lo, delta_sig_hi = delsigma_hi)


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    DEFAULTS = {}
    DEFAULTS['path2config'] = 'config/boss_lrg_xi.yaml'

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    main(**args)
