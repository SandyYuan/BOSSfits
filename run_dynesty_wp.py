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
from matplotlib.lines import Line2D
from matplotlib import rc, rcParams
rcParams.update({'font.size': 12})

from dynesty import NestedSampler
from dynesty import plotting as dyplot

from likelihood_boss import xirppi_Data, wp_Data
from abacusnbody.hod.abacus_hod import AbacusHOD

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/boss_lrg_wp.yaml'

def inrange(p, params):
    return np.all((p<=params[:, 3]) & (p>=params[:, 2]))

def lnprob(p, params, param_mapping, param_tracer, Data, Ball):
    # read the parameters 
    if inrange(p, params):
        for key in param_mapping.keys():
            mapping_idx = param_mapping[key]
            tracer_type = param_tracer[key]
            #tracer_type = param_tracer[params[mapping_idx, -1]]
            Ball.tracers[tracer_type][key] = p[mapping_idx]
            
        # we need to determine the expected number density 
        Ball.tracers['LRG']['ic'] = 1
        ngal_dict = Ball.compute_ngal(Nthread = 32)[0]
        # we are only dealing with lrgs here
        N_lrg = ngal_dict['LRG']
        # print(Ball.tracers['LRG']['ic'], N_lrg, Data.num_dens_mean['LRG']*Ball.params['Lbox']**3)
        Ball.tracers['LRG']['ic'] = \
            min(1, Data.num_dens_mean['LRG']*Ball.params['Lbox']**3/N_lrg)

        theory_density = {
        'LRG': N_lrg * Ball.tracers['LRG']['ic']/Ball.params['Lbox']**3
        }

        # pass them to the mock dictionary
        mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread = 64)

        clustering = Ball.compute_wp(mock_dict, Ball.rpbins, Ball.pimax, Ball.pi_bin_size, Nthread = 32)

        lnP = Data.compute_likelihood(clustering, theory_density)
    else:
        lnP = -np.inf
    return lnP

# prior transform function
def prior_transform(u, params_hod, params_hod_initial_range):
    return stats.norm.ppf(u, loc = params_hod, scale = params_hod_initial_range)

def main(path2config):

    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    dynesty_config_params = config['dynesty_config_params']
    fit_params = config['dynesty_fit_params']    
    
    # create a new abacushod object and load the subsamples
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

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

    # Make path to output
    if not os.path.isdir(os.path.expanduser(dynesty_config_params['path2output'])):
        try:
            os.makedirs(os.path.expanduser(dynesty_config_params['path2output']))
        except:
            pass
        
    # dynesty parameters
    nlive = dynesty_config_params['nlive']
    maxcall = dynesty_config_params['maxcall']
    dlogz = dynesty_config_params['dlogz']
    method = dynesty_config_params['method']
    bound = dynesty_config_params['bound']

    # where to record
    prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
                                dynesty_config_params['chainsPrefix'])

    # initiate sampler
    found_file = os.path.isfile(prefix_chain+'.dill')
    if (not found_file) or (not dynesty_config_params['rerun']):

        # initialize our nested sampler
        sampler = NestedSampler(lnprob, prior_transform, nparams, 
            logl_args = [params, param_mapping, param_tracer, newData, newBall], 
            ptform_args = [params[:, 0], params[:, 1]], 
            nlive=nlive, sample = method, rstate = np.random.RandomState(dynesty_config_params['rseed']))
            # first_update = {'min_eff': 20})

    else:
        # load sampler to continue the run
        with open(prefix_chain+'.dill', "rb") as f:
            sampler = dill.load(f)
        sampler.rstate = np.load(prefix_chain+'_results.npz', allow_pickle = True)['rstate']
    print("run sampler")

    sampler.run_nested(maxcall = maxcall, dlogz = dlogz)

    # save sampler itself
    with open(prefix_chain+'.dill', "wb") as f:
         dill.dump(sampler, f)
    res1 = sampler.results
    np.savez(prefix_chain+'_results.npz', res = res1, rstate = np.random.get_state())

def traceplots(path2config):
    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    dynesty_config_params = config['dynesty_config_params']
    fit_params = config['dynesty_fit_params']    

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

    # where to record
    prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
                                dynesty_config_params['chainsPrefix'])

    datafile = np.load(prefix_chain+'_results.npz', allow_pickle = True)
    res1 = datafile['res'].item()

    # print the max logl fit
    logls = res1['logl']
    indmax = np.argmax(logls)
    hod_params = res1['samples'][indmax]
    print("max logl fit ", hod_params, logls[indmax])

    # make plots 
    fig, axes = dyplot.runplot(res1)
    pl.tight_layout()
    fig.savefig('./plots_dynesty/runplot_'+dynesty_config_params['chainsPrefix']+'.pdf', dpi = 100)

    fig, axes = dyplot.traceplot(res1,
                             labels = list(fit_params.keys()),
                             truth_color='black', show_titles=True,
                             trace_cmap='viridis', 
                             trace_kwargs = {'edgecolor' : 'none'})
    pl.tight_layout()
    fig.savefig('./plots_dynesty/traceplot_'+dynesty_config_params['chainsPrefix']+'.pdf', dpi = 100)

    # plot initial run (res1; left)
    rcParams.update({'font.size': 12})
    # scatter corner plot
    fig, ax = dyplot.cornerpoints(res1, cmap = 'plasma', truths = list(params[:, 0]), labels = list(fit_params.keys()), kde = False)
    #pl.tight_layout()
    fig.savefig('./plots_dynesty/cornerpoints_'+dynesty_config_params['chainsPrefix']+'.pdf', dpi = 100)

    # scatter corner 
    fig, ax = dyplot.cornerplot(res1, color='blue', truths=hod_params,
                               # span = [(12.85, 13.05), (14, 14.5), (0, 0.5), (0.8, 1.2), (0, 1.0)],
                               truth_color='black', smooth = 0.04,
                               labels = ['$\log M_\mathrm{cut}$', '$\log M_1$', '$\sigma$', '$\\alpha$', '$\kappa$'],
                               show_titles=True, quantiles_2d = [0.393, 0.865, 0.989], 
                               max_n_ticks=5, quantiles=None, 
                               label_kwargs = {'fontsize' : 18},
                               hist_kwargs = {'histtype' : 'step'})
    fig.savefig('./plots_dynesty/cornerplot_'+dynesty_config_params['chainsPrefix']+'.pdf', dpi = 100)

    # corner with overlap
    cleanedfig, cleanedax = dyplot.cornerplot(res1, color='blue', truths=hod_params,
                               truth_color='blue', truth_kwargs = {'ls': '--'},
                               max_n_ticks=5, quantiles=None, smooth = 0.03,
                               label_kwargs = {'fontsize' : 18},
                               hist_kwargs = {'histtype' : 'step', 'density': True})
    #pl.tight_layout()
    # overplot uncleaned contours
    uncleaned_file = np.load('dynesty/test/wp_base_uncleaned_skip3bins_results.npz', allow_pickle = True)
    uncleaned_res = uncleaned_file['res'].item()
    uncleaned_logls = uncleaned_res['logl']
    uncleaned_indmax = np.argmax(uncleaned_logls)
    uncleaned_hod_params = uncleaned_res['samples'][uncleaned_indmax]

    fig, ax = dyplot.cornerplot(uncleaned_res, color='red', truths=uncleaned_hod_params,
                               span = [(12.85, 13.35), (14, 15.3), (0, 0.5), (0.4, 1.3), (0, 1.0)],
                               truth_color='red', truth_kwargs = {'ls': '--'},
                               labels = ['$\log M_\mathrm{cut}$', '$\log M_1$', '$\sigma$', '$\\alpha$', '$\kappa$'],
                               max_n_ticks=5, quantiles=None, smooth = 0.03,
                               label_kwargs = {'fontsize' : 18},
                               hist_kwargs = {'histtype' : 'step', 'density': True}, fig = (cleanedfig, cleanedax))
    ax[1][1].set_ylim(0, 7)
    ax[3][3].set_ylim(0, 6.5)
    ax[4][4].set_ylim(0, 2.6)

    custom_lines = [Line2D([0], [0], color='blue'),
                Line2D([0], [0], color='red')]
    pl.legend(custom_lines, ['cleaned', 'uncleaned'], bbox_to_anchor=(0.0, 3.5), fontsize = 15)
    fig.savefig('./plots_dynesty/cornerplot_'+dynesty_config_params['chainsPrefix']+'_overlap.pdf', dpi = 100)

def plot_bestfit(path2config):
    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    dynesty_config_params = config['dynesty_config_params']
    fit_params = config['dynesty_fit_params']    
    newData = wp_Data(data_params, HOD_params)
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

    # where to record
    prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
                                dynesty_config_params['chainsPrefix'])

    datafile = np.load(prefix_chain+'_results.npz', allow_pickle = True)
    res1 = datafile['res'].item()

    # print the max logl fit
    logls = res1['logl']
    indmax = np.argmax(logls)
    hod_params = res1['samples'][indmax]
    print(hod_params, logls[indmax])

    # read the parameters 
    for key in fit_params.keys():
        tracer_type = fit_params[key][-1]
        mapping_idx = fit_params[key][0]
        #tracer_type = param_tracer[params[mapping_idx, -1]]
        newBall.tracers[tracer_type][key] = hod_params[mapping_idx]

    newBall.tracers['LRG']['ic'] = 1
    ngal_dict = newBall.compute_ngal(Nthread = 32)[0]
    N_lrg = ngal_dict['LRG']
    newBall.tracers['LRG']['ic'] = min(1, newData.num_dens_mean['LRG']*newBall.params['Lbox']**3/N_lrg)

    print(newBall.tracers['LRG'])
    mock_dict = newBall.run_hod(newBall.tracers, newBall.want_rsd, Nthread = 64)
    Ncent = mock_dict['LRG']['Ncent']
    Ntot = len(mock_dict['LRG']['x'])
    print('satellite fraction ', (Ntot - Ncent)/Ntot)
    clustering = newBall.compute_wp(mock_dict, newBall.rpbins, newBall.pimax, newBall.pi_bin_size, Nthread = 16)

    rs = 0.5*(newBall.rpbins[1:] + newBall.rpbins[:-1])
    # plotting utility
    for etracer in clustering.keys():
        skipbins = 3
        lrgcov = np.linalg.inv(newData.icov[etracer])
        delta = (rs*clustering[etracer] - rs*newData.clustering[etracer])[skipbins:]
        print(clustering[etracer], newData.clustering[etracer])
        print(np.einsum('i,ij,j', delta, newData.icov[etracer][skipbins:, skipbins:], delta))

        fig, ax = pl.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [3, 1]}, figsize = (6,6))
        ax[0].plot(rs[skipbins:], rs[skipbins:]*clustering[etracer][skipbins:], label = "mock", alpha = 0.8)
        uncleaned = np.load("./data_wp_lrg_uncleaned_skip3bins.npz")
        delta = (uncleaned['rwp'] - rs*newData.clustering[etracer])[skipbins:]
        print("uncleaned, ", np.einsum('i,ij,j', delta, newData.icov[etracer][skipbins:, skipbins:], delta))
        # ax[0].plot(uncleaned['rs'][skipbins:], uncleaned['rwp'][skipbins:], label = "uncleaned", alpha = 0.8)
        ax[0].errorbar(rs[skipbins:], rs[skipbins:]*newData.clustering[etracer][skipbins:], yerr = np.sqrt(np.diagonal(lrgcov))[skipbins:], label = 'CMASS', alpha = 0.8)
        ax[0].set_xscale('log')
        ax[0].set_ylabel('$r w_p$ [($h^{-1}$Mpc)$^2$]')
        ax[0].legend(loc='best')

        ax[1].plot(rs[skipbins:], (clustering[etracer][skipbins:] - newData.clustering[etracer][skipbins:])/newData.clustering[etracer][skipbins:], label = "cleaned", alpha = 0.8)
        # ax[1].plot(uncleaned['rs'][skipbins:], (uncleaned['rwp'][skipbins:] - rs[skipbins:]*newData.clustering[etracer][skipbins:])/(rs[skipbins:]*newData.clustering[etracer][skipbins:]), label = "uncleaned", alpha = 0.8)
        ax[1].fill_between(rs[skipbins:], np.sqrt(np.diagonal(lrgcov))[skipbins:]/(rs[skipbins:]*newData.clustering[etracer][skipbins:]), 
            -np.sqrt(np.diagonal(lrgcov))[skipbins:]/(rs[skipbins:]*newData.clustering[etracer][skipbins:]), alpha = 0.4)
        ax[1].axhline(y = 0, xmin = 0, xmax = 1, ls = ':')
        ax[1].set_ylabel('$\delta w_p$')
        ax[1].set_xlabel('$r$ [$h^{-1}$Mpc]')

        pl.tight_layout()
        fig.savefig("./plots/plot_bestfit_"+dynesty_config_params['chainsPrefix']+"_"+etracer+".pdf", dpi = 200)
        # np.savez("./data_wp_lrg_cleaned_skip3bins", rs = rs, rwp = rs*clustering[etracer])
        np.savez("./plot_compare_data", x = rs[skipbins:], y1_uncleaned = uncleaned['rwp'][skipbins:],
                                                        y1_cleaned = rs[skipbins:]*newData.clustering[etracer][skipbins:],
                                                        y1_data_err = np.sqrt(np.diagonal(lrgcov))[skipbins:],
                                                        y1_data = rs[skipbins:]*newData.clustering[etracer][skipbins:], 
                                                        y2_uncleaned = (uncleaned['rwp'][skipbins:] - rs[skipbins:]*newData.clustering[etracer][skipbins:])/(rs[skipbins:]*newData.clustering[etracer][skipbins:]),
                                                        y2_cleaned = (clustering[etracer][skipbins:] - newData.clustering[etracer][skipbins:])/newData.clustering[etracer][skipbins:],
                                                        y2_shaded = np.sqrt(np.diagonal(lrgcov))[skipbins:]/(rs[skipbins:]*newData.clustering[etracer][skipbins:]))    

        # compare_data = np.load("./plot_compare_data.npz")
        # fig, ax = pl.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [3, 1]}, figsize = (6,6))
        # ax[0].plot(compare_data['x'], compare_data['y1_cleaned'], label = "cleaned", alpha = 0.8)
        # ax[0].plot(compare_data['x'], compare_data['y1_uncleaned'], label = "uncleaned", alpha = 0.8)
        # ax[0].errorbar(compare_data['x'], compare_data['y1_data'], yerr = compare_data['y1_data_err'], label = 'CMASS', alpha = 0.8)
        # ax[0].set_xscale('log')
        # ax[0].set_ylabel('$r w_p$ [($h^{-1}$Mpc)$^2$]')
        # ax[0].legend(loc='best')

        # ax[1].plot(compare_data['x'], compare_data['y2_cleaned'], label = "cleaned", alpha = 0.8)
        # ax[1].plot(compare_data['x'], compare_data['y2_uncleaned'], label = "uncleaned", alpha = 0.8)
        # ax[1].fill_between(compare_data['x'], compare_data['y2_shaded'], -compare_data['y2_shaded'], alpha = 0.4)
        # ax[1].axhline(y = 0, xmin = 0, xmax = 1, ls = ':')
        # ax[1].set_ylabel('$\delta w_p$')
        # ax[1].set_xlabel('$r$ [$h^{-1}$Mpc]')

        # pl.tight_layout()
        # fig.savefig("./plots/plot_bestfit_compare.pdf", dpi = 200)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    # main(**args)

    traceplots(**args)
    
    # plot_bestfit(**args)