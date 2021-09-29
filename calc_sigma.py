import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import rc, rcParams
rcParams.update({'font.size': 13})

import numpy as np
import os
import sys
import time
import yaml
import h5py
import argparse
from pathlib import Path
from astropy.table import Table
import numba
from numba import njit, types, jit
from numba.typed import List

from abacusnbody.data import read_abacus
from glob import glob
import astropy.table

from likelihood_boss import sigma_Data, wp_Data
from abacusnbody.hod.abacus_hod import AbacusHOD
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

import halotools
from halotools.mock_observables import mean_delta_sigma
from halotools.mock_observables import return_xyz_formatted_array

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/boss_lrg_xi.yaml'

np.random.seed(100)

def load_particles(simdir, simname, z_mock, downsample = 1):
    allp = []
    for fn in glob(simdir+simname+'/halos/z'+str(z_mock).ljust(5, '0')+'/*_rv_*/*.asdf'):
        start = time.time()
        newparts = read_abacus.read_asdf(fn, load_vel=False)
        if downsample < 1:
            sub_indices = np.random.choice(np.arange(len(newparts)), int(len(newparts)*downsample), replace = False)
            newparts = newparts[sub_indices]
        allp += [newparts]
        print(fn, len(newparts), time.time() - start)
    # concatenate into one big table
    allp = np.array(astropy.table.vstack(allp)['pos'])

    return allp


def calc_sigma(path2config, mock_key = 'LRG', combo_key = 'LRG_LRG', load_bestfit = True, downsample = 0.005):
    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    dynesty_config_params = config['dynesty_config_params']
    fit_params = config['dynesty_fit_params']    
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

    # load particles
    simname = config['sim_params']['sim_name'] # "AbacusSummit_base_c000_ph006"
    simdir = config['sim_params']['sim_dir']
    z_mock = config['sim_params']['z_mock']
    cleaned_halos = config['sim_params']['cleaned_halos']

    # particle subsample directory
    subsample_name = Path(sim_params['subsample_dir']) / simname / ('z%4.3f'%z_mock) / ('parts_all_down_%4.3f.h5'%downsample)
    if os.path.exists(subsample_name):
        newfile = h5py.File(subsample_name, 'r')
        full_subsample = np.array(newfile['particles'])
    else:
        start = time.time()
        full_subsample = load_particles(simdir, simname, z_mock, downsample)
        print(len(full_subsample), time.time() - start)

        newfile = h5py.File(subsample_name, 'w')
        dataset = newfile.create_dataset('particles', data = full_subsample)
        newfile.close()

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
    Mpart = newBall.params['Mpart'] # msun / h
    newData = sigma_Data(data_params, HOD_params)

    if load_bestfit:
        # # load chains
        # prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
        #                             dynesty_config_params['chainsPrefix'])

        # datafile = np.load(prefix_chain+'_results.npz', allow_pickle = True)
        # res1 = datafile['res'].item()

        # # print the max logl fit
        # logls = res1['logl']
        # indmax = np.argmax(logls)
        # hod_params = res1['samples'][indmax]
        # print("max logl fit ", hod_params, logls[indmax])
        hod_params = [13.00463651, 14.21416778, -5.21529117,  1.27676045, -0.25028219] # wp fit lowz
        # hod_params = [12.96536177, 14.12153545, -2.19574089,  1.20578504,  0.47029679,  0.16498357,
        # 0.86794809]
        # hod_params = [12.80287797, 13.99974778, -2.93816538,  1.03357034,  0.3651158,   0.17626678,
        # 1.00171066, -0.04554104, -0.16908941]
        # hod_params = [ 1.28350351e+01,  1.40692269e+01, -2.79634291e+00,  8.89787715e-01,
        #   9.55370398e-01,  1.74691015e-01,  1.03191577e+00, -1.26728445e-02, -3.74249020e-01]

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

    print(newBall.tracers['LRG'])
    mock_dict = newBall.run_hod(newBall.tracers, newBall.want_rsd, Nthread = 64)

    # get all the galaxy positions
    xgals = mock_dict[mock_key]['x']
    ygals = mock_dict[mock_key]['y']
    zgals = mock_dict[mock_key]['z']
    pos_gals = np.vstack((xgals, ygals, zgals)).T % newBall.lbox
    full_subsample = full_subsample  % newBall.lbox

    start = time.time()
    delta_sig = mean_delta_sigma(pos_gals, full_subsample, Mpart/0.03/downsample, 
        newData.rbins[mock_key+'_'+mock_key], newBall.lbox)
    print(time.time() - start, delta_sig)
    np.savez("./data_lensing/data_"+dynesty_config_params['chainsPrefix'], rs = newData.rs[combo_key], delta_sig = delta_sig)
    return newData.rs, delta_sig

def plot_sigma(path2config, mock_key = 'LRG', combo_key = 'LRG_LRG', load_bestfit = True, downsample = 0.005):
    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    dynesty_config_params = config['dynesty_config_params']
    fit_params = config['dynesty_fit_params']    

    newData = sigma_Data(data_params, HOD_params)

    mysigma = np.load("./data_lensing/data_"+dynesty_config_params['chainsPrefix']+".npz", allow_pickle = True)

    data_x = newData.rs[combo_key]
    data_y = newData.rs[combo_key] * newData.deltasigma[combo_key]
    data_yerr = newData.rs[combo_key] * np.sqrt(np.diag(newData.covs[combo_key]))

    fig, ax = pl.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [3, 1]}, figsize = (6,6))
    ax[0].plot(data_x, data_x*mysigma['delta_sig']/1e12, label = 'mock')
    ax[0].errorbar(data_x, data_y, yerr = data_yerr, label = 'observed')
    # ax[0].set_xlabel('$r_p$ ($h^{-1}$ Mpc)')
    ax[0].set_ylabel('$r_p \Delta \Sigma$ (Mpc $M_{\odot}$ pc$^{-2}$)')
    ax[0].set_xscale('log')
    ax[0].legend(loc='best')

    ax[1].plot(data_x, (data_x*mysigma['delta_sig']/1e12 - data_y)/data_y)
    ax[1].fill_between(data_x, data_yerr/data_y, -data_yerr/data_y, alpha = 0.4, color = 'C1')
    ax[1].axhline(y = 0, xmin = 0, xmax = 1, ls = ':', color = 'k')
    ax[1].set_xlabel('$r_p$ ($h^{-1}$ Mpc)')
    ax[1].set_ylabel('$\delta \Sigma$')

    pl.tight_layout()
    pl.savefig('./plots_lensing/plot_lensing_'+dynesty_config_params['chainsPrefix']+'.pdf', dpi = 200)

    # comparison plot
    sigma_xi_s_B = np.load("./data_lensing/data_xi_base_velbias_s_Bold_nlive500_cleaned_logsigma_skipcol.npz", allow_pickle = True)
    sigma_xi_B = np.load("./data_lensing/data_xi_base_velbias_Bold_nlive500_cleaned_logsigma_skipcol.npz", allow_pickle = True)
    sigma_xi_B_err = np.load("./data_lensing/data_xi_base_velbias_Bold_nlive500_cleaned_logsigma_skipcol_werror.npz")
    sigma_xi = np.load("./data_lensing/data_xi_base_velbias_nlive500_cleaned_logsigma_skipcol.npz", allow_pickle = True)
    sigma_wp = np.load("./data_lensing/data_wp_base_cleaned_skip3bins.npz", allow_pickle = True)
    # sigma_xi_B5r = np.load("./data_lensing/data_xi_base_velbias_Bold5r_nlive500_cleaned_logsigma_skipcol.npz", allow_pickle = True)

    sig_des = Table.read('../ggl/des_0.csv')
    sig_hsc = Table.read('../ggl/hsc_0.csv')
    sig_kids = Table.read('../ggl/kids_0.csv')
    sig_new = np.load('../ggl/data_ds_comb.npz')
    fig, ax = pl.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [3, 1]}, figsize = (5.2,6))
    ax[0].plot(data_x, data_x*sigma_wp['delta_sig']/1e12, color = 'C3', label = '$w_p$ fit')
    ax[0].plot(data_x, data_x*sigma_xi['delta_sig']/1e12, color = 'C1', label = '$\\xi$ fit')
    ax[0].plot(data_x, data_x*sigma_xi_B['delta_sig']/1e12, color = 'C4', label = 'include $B$')
    ax[0].fill_between(data_x, data_x*sigma_xi_B_err['delta_sig_hi']/1e12, data_x*sigma_xi_B_err['delta_sig_lo']/1e12, alpha = 0.4, color = 'C4')
    ax[0].plot(data_x, data_x*sigma_xi_s_B['delta_sig']/1e12, color = 'C4', ls = '--', label = 'include $s+B$')
    # ax[0].plot(data_x, data_x*sigma_xi_B5r['delta_sig']/1e12, color = 'C4', ls = ':', label = 'include $B 5r$')
    ax[0].errorbar(data_x, data_y, yerr = data_yerr, label = 'observed', color = 'C0', marker = 'o')
    # ax[0].plot(sig_des['rp'], sig_des['rp']*sig_des['ds'], marker = 'o', label = 'DES')
    # ax[0].plot(sig_hsc['rp'], sig_hsc['rp']*sig_hsc['ds'], marker = '+', label = 'HSC')
    # ax[0].plot(sig_kids['rp'], sig_kids['rp']*sig_kids['ds'], marker = '^', label = 'KiDS')
    # ax[0].errorbar(sig_new['rp'], sig_new['rp']*sig_new['ds'], yerr = sig_new['rp']*sig_new['ds_err'], 
        # marker = '^', color = 'C5', label = 'DES+HSC+KiDS')
    ax[0].set_ylabel('$r_p \Delta \Sigma$ (Mpc $M_{\odot}$ pc$^{-2}$)')
    ax[0].set_ylim(2, 10)
    ax[0].set_xscale('log')
    ax[0].legend(loc='best', fontsize = 9)

    ax[1].plot(data_x, (data_x*sigma_wp['delta_sig']/1e12 - data_y)/data_y, color = 'C3')
    ax[1].plot(data_x, (data_x*sigma_xi['delta_sig']/1e12 - data_y)/data_y, color = 'C1')
    ax[1].plot(data_x, (data_x*sigma_xi_B['delta_sig']/1e12 - data_y)/data_y, color = 'C4')
    ax[1].plot(data_x, (data_x*sigma_xi_s_B['delta_sig']/1e12 - data_y)/data_y, color = 'C4', ls = '--')
    # ax[1].plot(data_x, (data_x*sigma_xi_B5r['delta_sig']/1e12 - data_y)/data_y, color = 'C4', ls = ':')
    ax[1].fill_between(data_x, data_yerr/data_y, -data_yerr/data_y, alpha = 0.4, color = 'C0')
    ax[1].axhline(y = 0, xmin = 0, xmax = 1, ls = ':', color = 'C0')
    ax[1].set_xlabel('$r_p$ ($h^{-1}$ Mpc)')
    ax[1].set_ylabel('$\delta \Sigma$')

    pl.tight_layout()
    pl.savefig('./plots_lensing/plot_lensing_compare_Bold.pdf', dpi = 200)

    # comparison plot
    sigma_xi_s_A = np.load("./data_lensing/data_xi_base_velbias_s_A_nlive500_cleaned_logsigma_skipcol.npz", allow_pickle = True)
    sigma_xi_A = np.load("./data_lensing/data_xi_base_velbias_A_nlive500_cleaned_logsigma_skipcol.npz", allow_pickle = True)
    sigma_xi_A_err = np.load("./data_lensing/data_xi_base_velbias_A_nlive500_cleaned_logsigma_skipcol_werror.npz", allow_pickle = True)
    sigma_xi = np.load("./data_lensing/data_xi_base_velbias_nlive500_cleaned_logsigma_skipcol.npz", allow_pickle = True)
    sigma_wp = np.load("./data_lensing/data_wp_base_cleaned_skip3bins.npz", allow_pickle = True)

    fig, ax = pl.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [3, 1]}, figsize = (5.2,6))
    ax[0].plot(data_x, data_x*sigma_wp['delta_sig']/1e12, color = 'C3', label = '$w_p$ fit')
    ax[0].plot(data_x, data_x*sigma_xi['delta_sig']/1e12, color = 'C1', label = '$\\xi$ fit')
    ax[0].plot(data_x, data_x*sigma_xi_A['delta_sig']/1e12, color = 'C4', label = 'include $A$')
    ax[0].fill_between(data_x, data_x*sigma_xi_A_err['delta_sig_hi']/1e12, data_x*sigma_xi_A_err['delta_sig_lo']/1e12, alpha = 0.4, color = 'C4')
    ax[0].plot(data_x, data_x*sigma_xi_s_A['delta_sig']/1e12, color = 'C4', ls = '--', label = 'include $s+A$')
    ax[0].errorbar(data_x, data_y, yerr = data_yerr, label = 'observed', color = 'C0', marker = 'o')
    ax[0].set_ylabel('$r_p \Delta \Sigma$ (Mpc $M_{\odot}$ pc$^{-2}$)')
    ax[0].set_ylim(2, 10)
    ax[0].set_xscale('log')
    ax[0].legend(loc='best', fontsize = 9)

    ax[1].plot(data_x, (data_x*sigma_wp['delta_sig']/1e12 - data_y)/data_y, color = 'C3')
    ax[1].plot(data_x, (data_x*sigma_xi['delta_sig']/1e12 - data_y)/data_y, color = 'C1')
    ax[1].plot(data_x, (data_x*sigma_xi_A['delta_sig']/1e12 - data_y)/data_y, color = 'C4')
    ax[1].plot(data_x, (data_x*sigma_xi_s_A['delta_sig']/1e12 - data_y)/data_y, color = 'C4', ls = '--')
    ax[1].fill_between(data_x, data_yerr/data_y, -data_yerr/data_y, alpha = 0.4, color = 'C0')
    ax[1].axhline(y = 0, xmin = 0, xmax = 1, ls = ':', color = 'C0')
    ax[1].set_xlabel('$r_p$ ($h^{-1}$ Mpc)')
    ax[1].set_ylabel('$\delta \Sigma$')

    pl.tight_layout()
    pl.savefig('./plots_lensing/plot_lensing_compare_A.pdf', dpi = 200)


    # comparison plot
    sigma_multi_B = np.load("./data_lensing/data_wpmultipoles_base_velbias_Bold_nlive500_cleaned_logsigma_skipcol_1.npz", allow_pickle = True)
    sigma_multi_A = np.load("./data_lensing/data_wpmultipoles_base_velbias_A_nlive500_cleaned_logsigma_skipcol.npz", allow_pickle = True)
    sigma_wp = np.load("./data_lensing/data_wp_base_cleaned_skip3bins.npz", allow_pickle = True)
    # sigma_xi_B5r = np.load("./data_lensing/data_xi_base_velbias_Bold5r_nlive500_cleaned_logsigma_skipcol.npz", allow_pickle = True)

    fig, ax = pl.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [3, 1]}, figsize = (5.2,6))
    ax[0].plot(data_x, data_x*sigma_wp['delta_sig']/1e12, color = 'C3', label = '$w_p$ fit')
    ax[0].plot(data_x, data_x*sigma_multi_B['delta_sig']/1e12, color = 'C1', label = 'multipole fit $+B$')
    ax[0].plot(data_x, data_x*sigma_multi_A['delta_sig']/1e12, color = 'C4', label = 'multipole fit $+A$')
    # ax[0].plot(data_x, data_x*sigma_xi_B5r['delta_sig']/1e12, color = 'C4', ls = ':', label = 'include $B 5r$')
    ax[0].errorbar(data_x, data_y, yerr = data_yerr, label = 'observed', color = 'C0', marker = 'o')
    ax[0].set_ylabel('$r_p \Delta \Sigma$ (Mpc $M_{\odot}$ pc$^{-2}$)')
    ax[0].set_ylim(2, 10)
    ax[0].set_xscale('log')
    ax[0].legend(loc='best', fontsize = 9)

    ax[1].plot(data_x, (data_x*sigma_wp['delta_sig']/1e12 - data_y)/data_y, color = 'C3')
    ax[1].plot(data_x, (data_x*sigma_multi_B['delta_sig']/1e12 - data_y)/data_y, color = 'C1')
    ax[1].plot(data_x, (data_x*sigma_multi_A['delta_sig']/1e12 - data_y)/data_y, color = 'C4')
    # ax[1].plot(data_x, (data_x*sigma_xi_B5r['delta_sig']/1e12 - data_y)/data_y, color = 'C4', ls = ':')
    ax[1].fill_between(data_x, data_yerr/data_y, -data_yerr/data_y, alpha = 0.4, color = 'C0')
    ax[1].axhline(y = 0, xmin = 0, xmax = 1, ls = ':', color = 'C0')
    ax[1].set_xlabel('$r_p$ ($h^{-1}$ Mpc)')
    ax[1].set_ylabel('$\delta \Sigma$')

    pl.tight_layout()
    pl.savefig('./plots_lensing/plot_lensing_compare_multi_1.pdf', dpi = 200)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    calc_sigma(**args)
    # plot_sigma(**args)

