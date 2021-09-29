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

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


DEFAULTS = {}
DEFAULTS['path2config'] = 'config/boss_lrg_multipoles.yaml'

def inrange(p, params):
    return np.all((p<=params[:, 3]) & (p>=params[:, 2]))

def lnprob(p, params, param_mapping, param_tracer, Data, Ball):
    if inrange(p, params):
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

        clustering = Ball.compute_multipole(mock_dict, Ball.rpbins, Ball.pimax, Nthread = 16)

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
    newData = multipole_Data(data_params, HOD_params)

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
        sampler.rstate = np.random.RandomState(dynesty_config_params['rseed']) # np.load(prefix_chain+'_results.npz', allow_pickle = True)['rstate']
    print("run sampler")

    sampler.run_nested(maxcall = maxcall, dlogz = dlogz)

    # save sampler itself
    with open(prefix_chain+'.dill', "wb") as f:
         dill.dump(sampler, f)
    res1 = sampler.results
    np.savez(prefix_chain+'_results.npz', res = res1, rstate = np.random.get_state())

def bestfit_logl(path2config):

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
    newData = multipole_Data(data_params, HOD_params)

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

    res1['samples'][:, 5] = abs(res1['samples'][:, 5])

    # print the max logl fit
    logls = res1['logl']
    indmax = np.argmax(logls)
    hod_params = res1['samples'][indmax]

    print("bestfit logl", lnprob(hod_params, params, param_mapping, param_tracer, newData, newBall))

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

    res1['samples'][:, 5] = abs(res1['samples'][:, 5])

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
                               truth_color='black', 
                               labels = list(fit_params.keys()), # ['$\log M_\mathrm{cut}$', '$\log M_1$', '$\log \sigma$', '$\\alpha$', '$\kappa$', '$\\alpha_c$', '$\\alpha_s$'], 
                               show_titles=True,
                               max_n_ticks=5, quantiles=None, 
                               label_kwargs = {'fontsize' : 18},
                               hist_kwargs = {'histtype' : 'step'})
    #pl.tight_layout()
    fig.savefig('./plots_dynesty/cornerplot_'+dynesty_config_params['chainsPrefix']+'.pdf', dpi = 100)

    # # corner with overlap
    # cleanedfig, cleanedax = dyplot.cornerplot(res1, color='blue', truths=hod_params,
    #                            truth_color='blue', truth_kwargs = {'ls': '--'},
    #                            max_n_ticks=5, quantiles=None, smooth = 0.03,
    #                            label_kwargs = {'fontsize' : 18},
    #                            hist_kwargs = {'histtype' : 'step', 'density': True})
    # #pl.tight_layout()
    # # overplot uncleaned contours
    # uncleaned_file = np.load('dynesty/test/xi_base_velbias_nlive500_results.npz', allow_pickle = True)
    # uncleaned_res = uncleaned_file['res'].item()
    # uncleaned_logls = uncleaned_res['logl']
    # uncleaned_indmax = np.argmax(uncleaned_logls)
    # uncleaned_hod_params = uncleaned_res['samples'][uncleaned_indmax]

    # fig, ax = dyplot.cornerplot(uncleaned_res, color='red', truths=uncleaned_hod_params,
    #                            span = [(12.8, 13.0), (14, 14.5), (0, 0.25), (0.6, 1.2), (0, 1.0), (0.1, 0.4), (0.8, 1.3)],
    #                            truth_color='red', truth_kwargs = {'ls': '--'},
    #                            labels = ['$\log M_\mathrm{cut}$', '$\log M_1$', '$\sigma$', '$\\alpha$', '$\kappa$', '$\\alpha_c$', '$\\alpha_s$'],
    #                            max_n_ticks=5, quantiles=None, smooth = 0.03,
    #                            label_kwargs = {'fontsize' : 18},
    #                            hist_kwargs = {'histtype' : 'step', 'density': True}, fig = (cleanedfig, cleanedax))
    # ax[1][1].set_ylim(0, 7)
    # ax[3][3].set_ylim(0, 6.5)
    # ax[4][4].set_ylim(0, 2.6)

    # custom_lines = [Line2D([0], [0], color='blue'),
    #             Line2D([0], [0], color='red')]
    # pl.legend(custom_lines, ['cleaned', 'uncleaned'], bbox_to_anchor=(0.0, 3.5), fontsize = 15)
    # fig.savefig('./plots_dynesty/cornerplot_'+dynesty_config_params['chainsPrefix']+'_overlap.pdf', dpi = 100)


def plot_bestfit(path2config):
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

    # res1['samples'][:, 5] = abs(res1['samples'][:, 5])

    # print the max logl fit
    logls = res1['logl']
    indmax = np.argmax(logls)
    hod_params = res1['samples'][indmax]
    print("max logl fit ", hod_params, logls[indmax])

    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    newData = xirppi_Data(data_params, HOD_params)

    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        tracer_type = param_tracer[key]
        #tracer_type = param_tracer[params[mapping_idx, -1]]
        if key == 'sigma':
            newBall.tracers[tracer_type][key] = 10**hod_params[mapping_idx]
        else:
            newBall.tracers[tracer_type][key] = hod_params[mapping_idx] 

    # we need to determine the expected number density 
    newBall.tracers['LRG']['ic'] = 1
    ngal_dict = newBall.compute_ngal(Nthread = 32)[0]
    # we are only dealing with lrgs here
    N_lrg = ngal_dict['LRG']
    print("Nlrg ", N_lrg, "data density ", newData.num_dens_mean['LRG'])
    # print(Ball.tracers['LRG']['ic'], N_lrg, Data.num_dens_mean['LRG']*Ball.params['Lbox']**3)
    newBall.tracers['LRG']['ic'] = \
        min(1, newData.num_dens_mean['LRG']*newBall.params['Lbox']**3/N_lrg)
    print("fic", newBall.tracers['LRG']['ic'])
    mock_dict = newBall.run_hod(newBall.tracers, newBall.want_rsd, Nthread = 64)
    Ncent = mock_dict['LRG']['Ncent']
    Ntot = len(mock_dict['LRG']['x'])
    print('satellite fraction ', (Ntot - Ncent)/Ntot)

    clustering = newBall.compute_xirppi(mock_dict, newBall.rpbins, newBall.pimax, newBall.pi_bin_size, Nthread = 16)
    
    key = 'LRG_LRG'
    mock_xi = clustering[key]
    data_xi = newData.clustering[key].reshape(np.shape(mock_xi))
    delta_xi_norm = (mock_xi - data_xi) / newData.diag[key].reshape(np.shape(mock_xi))

    # make a triple plot, xi, delta xi, chi2
    fig = pl.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(ncols = 3, nrows = 2, width_ratios = [1, 1, 1], height_ratios = [1, 12]) 
    mycmap2 = cm.get_cmap('bwr')

    # plot 1
    ax1 = fig.add_subplot(gs[3])
    ax1.set_xlabel('$\log r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax1.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    col1 = ax1.imshow(mock_xi.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(newBall.rpbins[0]), np.log10(newBall.rpbins[-1]), 0, newBall.pimax], 
        cmap = cm.viridis, norm=colors.LogNorm(vmin = 0.01, vmax = 30))
    ax1.set_yticks(np.linspace(0, 30, 7))
    # ax1.set_yticklabels(pibins)

    ax0 = fig.add_subplot(gs[0])
    cbar = pl.colorbar(col1, cax = ax0, orientation="horizontal")
    cbar.set_label('$\\xi(r_p, \pi)$', labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')

    # plot 2
    print(delta_xi_norm)
    ax2 = fig.add_subplot(gs[4])
    ax2.set_xlabel('$\log r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax2.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    col2 = ax2.imshow(delta_xi_norm.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(newBall.rpbins[0]), np.log10(newBall.rpbins[-1]), 0, newBall.pimax], 
        cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=-10, vmax=10))
    ax2.set_yticks(np.linspace(0, 30, 7))
    # ax2.set_yticklabels(pibins)

    ax3 = fig.add_subplot(gs[1])
    cbar = pl.colorbar(col2, cax = ax3, orientation="horizontal", ticks = [-10, -5, 0, 5, 10])
    cbar.set_label("$(\\xi_{\\rm{mock}}-\\xi_{\\rm{BOSS}})/\sigma(\\xi)$", labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')
    # cbar.set_ticks(np.linspace(-1, 1, num = 5))

    chi2s = (delta_xi_norm.flatten()[6:] * np.dot(newData.icov[key][6:, 6:], delta_xi_norm.flatten()[6:]))
    newshape = (np.shape(delta_xi_norm)[0]-1, np.shape(delta_xi_norm)[1])
    chi2s = chi2s.reshape(newshape)
    print(chi2s, np.sum(chi2s))
    ax2 = fig.add_subplot(gs[5])
    ax2.set_xlabel('$\log r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax2.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    col2 = ax2.imshow(chi2s.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(newBall.rpbins[0]), np.log10(newBall.rpbins[-1]), 0, newBall.pimax], 
        cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=-100, vmax=100))
    ax2.set_yticks(np.linspace(0, 30, 7))
    # ax2.set_yticklabels(pibins)

    ax3 = fig.add_subplot(gs[2])
    cbar = pl.colorbar(col2, cax = ax3, orientation="horizontal")
    cbar.set_label("$\chi^2$", labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')
    # cbar.set_ticks(np.linspace(-1, 1, num = 5))

    pl.subplots_adjust(wspace=20)
    pl.tight_layout()
    fig.savefig("./plots_dynesty/plot_bestfit_"+dynesty_config_params['chainsPrefix']+".pdf", dpi=720)



class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    main(**args)

    # traceplots(**args)
    # bestfit_logl(**args)
    # plot_bestfit(**args)
