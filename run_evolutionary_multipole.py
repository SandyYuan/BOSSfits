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
from matplotlib import gridspec
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from matplotlib import rc, rcParams
rcParams.update({'font.size': 10})

from stochopy import MonteCarlo, Evolutionary

from dynesty import NestedSampler
from dynesty import plotting as dyplot

from likelihood_boss import multipole_Data
from abacusnbody.hod.abacus_hod import AbacusHOD

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/boss_lrg_multipoles.yaml'

def inrange(p, params):
    return np.all((p<=params[:, 3]) & (p>=params[:, 2]))

def chi2(p, param_mapping, param_tracer, Data, Ball):
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

    print("ball rpbins", Ball.rpbins)
    start = time.time()
    clustering = Ball.compute_multipole(mock_dict, Ball.rpbins, Ball.pimax, Nthread = 32)
    print("clustering took time", time.time() - start)
    lnP = Data.compute_likelihood(clustering, theory_density)

    return -2*lnP

def main(path2config):

    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    optimize_config_params = config['optimize_config_params']
    fit_params = config['optimize_params']    
    
    # create a new abacushod object and load the subsamples
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

    # read data parameters
    newData = multipole_Data(data_params, HOD_params)

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
        
    # pbest = [12.80287797, 13.99974778, -2.93816538,  1.03357034,  0.3651158,   0.17626678,
    #     1.00171066, -0.04554104, -0.16908941]
    # chi2(pbest, param_mapping, param_tracer, newData, newBall)
    # print("done best fit")
    # where to record
    prefix_chain = os.path.join(os.path.expanduser(optimize_config_params['path2output']),
                                optimize_config_params['chainsPrefix'])

    popsize = optimize_config_params['popsize']
    max_iter = optimize_config_params['max_iter']
    ftol = optimize_config_params['ftol']

    ea = Evolutionary(chi2, args = [param_mapping, param_tracer, newData, newBall], 
        lower = params[:, 0], upper = params[:, 1], popsize = popsize, max_iter = max_iter)
    xopt, gfit = ea.optimize(solver = optimize_config_params['solver'])
    print(xopt, gfit)

def plot_bestfit(path2config, params = None, load_bestfit = True, tracer_pair = 'LRG_LRG'):

    # load the yaml parameters
    config = yaml.load(open(path2config))
    # update params if needed
    if params is None:
        params = {}
    config.update(params)

    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    fit_params = config['dynesty_fit_params']    
    newData = multipole_Data(data_params, HOD_params)
    
    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    write_to_disk = HOD_params['write_to_disk']
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1)
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']
    
    # create a new abacushod object
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    # print("mf", tot_mf)
    if load_bestfit:
        # acess best fit hod
        dynesty_config_params = config['dynesty_config_params']
        prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
                                    dynesty_config_params['chainsPrefix'])

        # datafile = np.load(prefix_chain+'_results.npz', allow_pickle = True)
        # res1 = datafile['res'].item()

        # # print the max logl fit
        # logls = res1['logl']
        # indmax = np.argmax(logls)
        # hod_params = res1['samples'][indmax]
        # print("best fit", hod_params, logls[indmax])
        # hod_params = [12.80287797, 13.99974778, -2.93816538,  1.03357034,  0.3651158,   0.17626678,
        #     1.00171066, -0.04554104, -0.16908941] # Bold fit
        hod_params = [12.70525188, 13.80426605, -2.79409949,  0.99480044 , 1.,          0.20628246,
                 1.00506506, -0.19501842,  0.31357795]
        for key in fit_params.keys():
            tracer_type = fit_params[key][-1]
            mapping_idx = fit_params[key][0]
            #tracer_type = param_tracer[params[mapping_idx, -1]]
            # newBall.tracers[tracer_type][key] = hod_params[mapping_idx]
            if key == 'sigma':
                newBall.tracers[tracer_type][key] = 10**hod_params[mapping_idx]
            else:
                newBall.tracers[tracer_type][key] = hod_params[mapping_idx] 
            print(key, newBall.tracers[tracer_type][key])

    newBall.tracers['LRG']['ic'] = 1
    ngal_dict = newBall.compute_ngal(Nthread = 32)[0]
    N_lrg = ngal_dict['LRG']
    newBall.tracers['LRG']['ic'] = min(1, newData.num_dens_mean['LRG']*newBall.params['Lbox']**3/N_lrg)
    
    mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk = False, Nthread = 64)
    clustering = newBall.compute_multipole(mock_dict, newBall.rpbins, newBall.pimax, Nthread = 32)

    r = newData.rs[tracer_pair]
    fig = pl.figure(figsize=(8.5, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios = [1, 1], height_ratios = [1, 1]) 

    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlabel('$r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax1.set_ylabel('$r_p w_p$ ($h^{-2} \mathrm{Mpc}^2)$')
    ax1.errorbar(r, r*newData.multipoles[tracer_pair][:8], yerr = r*newData.diag[tracer_pair][:8], label = 'observed')
    ax1.plot(r, r*clustering[tracer_pair][:8], label = 'mock')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(0.1, 50)
    ax1.legend(loc='best', prop={'size': 13})

    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlabel('$r$ ($h^{-1} \mathrm{Mpc}$)')
    ax2.set_ylabel('$r^2\\xi_0(r)$')
    ax2.set_xscale('log')
    # ax2.set_yscale('log')
    ax2.errorbar(r, newData.multipoles[tracer_pair][8:16]*r**2, yerr = newData.diag[tracer_pair][8:16]*r**2, label = "observed")
    ax2.plot(r, clustering[tracer_pair][8:16]*r**2, label = 'mock')
    ax2.set_xlim(0.1, 50)
    # ax2.legend(loc='best', prop={'size': 13})

    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlabel('$r$ ($h^{-1} \mathrm{Mpc}$)')
    ax3.set_ylabel('$r^2\\xi_2(r)$')
    ax3.set_xscale('log')
    # ax3.set_yscale('log')
    ax3.errorbar(r, newData.multipoles[tracer_pair][16:24]*r**2, yerr = newData.diag[tracer_pair][16:24]*r**2, label = "observed")
    ax3.plot(r, clustering[tracer_pair][16:24]*r**2, label = 'mock')
    ax3.set_xlim(0.1, 50)

    ax4 = fig.add_subplot(gs[3])
    ax4.set_xlabel('$r$ ($h^{-1} \mathrm{Mpc}$)')
    ax4.set_ylabel('$r^2 \\xi_4(r)$')
    ax4.set_xscale('log')
    # ax3.set_yscale('log')
    ax4.errorbar(r, newData.multipoles[tracer_pair][24:32]*r**2, yerr = newData.diag[tracer_pair][24:32]*r**2, label = "observed")
    ax4.plot(r, clustering[tracer_pair][24:32]*r**2, label = 'mock')
    ax4.set_xlim(0.1, 50)

    # ax5 = fig.add_subplot(gs[4])
    # ax5.set_xlabel('$r$ ($h^{-1} \mathrm{Mpc}$)')
    # ax5.set_ylabel('$r^2 \\xi_6(r)$')
    # ax5.set_xscale('log')
    # # ax3.set_yscale('log')
    # ax5.errorbar(rs, xi6_hong*rs**2, yerr = xi6_err*rs**2, label = "observed")
    # ax5.errorbar(rs, xi6_avg*rs**2, yerr = xi6_std*rs**2, label = 'mock')
    # ax5.set_xlim(0.1, 50)

    # ax6 = fig.add_subplot(gs[5])
    # ax6.set_xlabel('$r$ ($h^{-1} \mathrm{Mpc}$)')
    # ax6.set_ylabel('$r^2 \\xi_8(r)$')
    # ax6.set_xscale('log')
    # # ax3.set_yscale('log')
    # ax6.errorbar(rs, xi8_hong*rs**2, yerr = xi8_err*rs**2, label = "observed")
    # ax6.errorbar(rs, xi8_avg*rs**2, yerr = xi8_std*rs**2, label = 'mock')
    # ax6.set_xlim(0.1, 50)

    pl.tight_layout()
    fig.savefig("./plots/plot_multipoles_"+dynesty_config_params['chainsPrefix']+".pdf", dpi=720)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    # main(**args)
    plot_bestfit(**args)
