
import os
from pathlib import Path
import yaml

import numpy as np
import random
import time
from astropy.table import Table
import h5py
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator
from itertools import repeat
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import rc, rcParams
rcParams.update({'font.size': 11})

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from abacusnbody.hod.GRAND_HOD import *
from abacusnbody.hod.abacus_hod import AbacusHOD
from likelihood_boss import xirppi_Data, wp_Data

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/abacus_hod.yaml'

def load_slab(i, simdir, simname, z_mock, cleaning, log_mbins):
    print(i)
    cat = CompaSOHaloCatalog(
        simdir+simname+'/halos/z'+str(z_mock).ljust(5, '0')+'/halo_info/halo_info_'\
        +str(i).zfill(3)+'.asdf', fields = ['N'], cleaned_halos = cleaning)
    halos = cat.halos
    header = cat.header
    Mpart = header['ParticleMassHMsun'] # msun / h 
    H0 = header['H0']

    halo_mass = halos['N']*Mpart

    massfunction, edges = np.histogram(np.log10(halo_mass), bins = log_mbins)
    log_mmids = 0.5*(log_mbins[1:] + log_mbins[:-1])

    return massfunction, log_mmids

def halo_mfunc(path2config):
    config = yaml.load(open(path2config))

    sim_params = config['sim_params']
    fit_params = config['dynesty_fit_params']    
    clustering_params = config['clustering_params']
    HOD_params = config['HOD_params']
    data_params = config['data_params']
    simname = sim_params['sim_name'] # "AbacusSummit_base_c000_ph006"
    simdir = sim_params['sim_dir']
    z_mock = sim_params['z_mock']
    cleaned_halos = sim_params['cleaned_halos']
    halo_info_fns = \
    list((Path(simdir) / Path(simname) / 'halos' / ('z%4.3f'%z_mock) / 'halo_info').glob('*.asdf'))
    numslabs = len(halo_info_fns)

    # compute halo mass function
    log_mbins = np.linspace(12, 15.5, 81)
    tot_mf = 0
    for i in range(numslabs):
        new_mf, logms = load_slab(i, simdir, simname, z_mock, cleaned_halos, log_mbins)
        tot_mf += new_mf
    fn = "./data/data_nM_"+simname+"_z"+str(z_mock)
    if cleaned_halos:
        fn += "_cleaned"
    else:
        fn += "_uncleaned"
    np.savez(fn, logm_bins = log_mbins, nM = tot_mf)

def plot_mfunc(path2config):
    config = yaml.load(open(path2config))

    sim_params = config['sim_params']
    fit_params = config['dynesty_fit_params']    
    clustering_params = config['clustering_params']
    HOD_params = config['HOD_params']
    data_params = config['data_params']
    simname = sim_params['sim_name'] # "AbacusSummit_base_c000_ph006"
    simdir = sim_params['sim_dir']
    z_mock = sim_params['z_mock']
    cleaned_halos = sim_params['cleaned_halos']# calculate total galaxy counts for the fiducial HOD

    fn = "./data/data_nM_"+simname+"_z"+str(z_mock)
    if cleaned_halos:
        fn += "_cleaned"
    else:
        fn += "_uncleaned"

    summit_file = np.load(fn + ".npz")
    x1 = 0.5*(summit_file['logm_bins'][1:] + summit_file['logm_bins'][:-1])
    y1 = summit_file['nM']/(2000**3)
    s1_inv_neg = interpolate.InterpolatedUnivariateSpline(-np.log10(y1[10:-10]), x1[10:-10])
    s1 = interpolate.InterpolatedUnivariateSpline(x1[:-10], np.log10(y1[:-10]))
    cosmos_file = np.load("../s3PCF_fenv/data/data_avgnM_halos_ac_wider_1100.npz") # per 1100 box
    x2 = 0.5*(np.log10(cosmos_file['mbins'][1:]) + np.log10(cosmos_file['mbins'][:-1]))
    y2 = cosmos_file['nM']/(1100**3)
    s2_inv_neg = interpolate.InterpolatedUnivariateSpline(-np.log10(y2[11:-10]), x2[11:-10])
    s2 = interpolate.InterpolatedUnivariateSpline(x2[10:-10], np.log10(y2[10:-10]))

    fig = pl.figure(figsize = (5, 4))
    pl.plot(x1, y1, alpha = 0.7, label = 'summit')
    pl.plot(x2, y2, alpha = 0.7, label = 'cosmos')
    pl.axvline(x = np.log10(4e12 * 0.69), ymin = 0, ymax = 1, ls = '--', alpha = 0.5)
    pl.yscale('log')
    pl.legend(loc = 'best')
    pl.xlabel('$\log M_h$')
    pl.ylabel('$N_h$')
    pl.tight_layout()
    fig.savefig("./plots/plot_nM_compare.pdf", dpi = 200)

    fig = pl.figure(figsize = (5, 4))
    xs = np.linspace(12.6, 14.5, 20)
    y1_pred_log = s1(xs)
    xs_pred = s2_inv_neg(-y1_pred_log)
    delta_xs = xs_pred - xs 
    pl.plot(xs, delta_xs, alpha = 0.7, label = 'summit')
    pl.legend(loc = 'best')
    pl.xlabel('$\log M_h$')
    pl.ylabel('$\log M_\mathrm{cosmos} - \log M_\mathrm{summit}$')
    pl.tight_layout()
    fig.savefig("./plots/plot_deltaM.pdf", dpi = 200)

def calc_Ngal_tot(path2config):

    print("compute expected number of LRGs for a sim, subsampling, no assmbly bias")

    config = yaml.load(open(path2config))

    simname = config['sim_params']['sim_name'] # "AbacusSummit_base_c000_ph006"
    simdir = config['sim_params']['sim_dir']
    z_mock = config['sim_params']['z_mock']
    cleaned_halos = config['sim_params']['cleaned_halos']
    halo_info_fns = \
    list((Path(simdir) / Path(simname) / 'halos' / ('z%4.3f'%z_mock) / 'halo_info').glob('*.asdf'))
    numslabs = len(halo_info_fns)

    log_mbins = np.linspace(12, 15.589051329048312, 201)
    tot_mf = 0
    for i in range(numslabs):
        new_mf, logms = load_slab(i, simdir, simname, z_mock, cleaned_halos, log_mbins)
        tot_mf += new_mf

    print(tot_mf)
    # compute number of tracers
    tracer_flags = config['HOD_params']['tracer_flags']

    for tracer in tracer_flags.keys():
        if tracer_flags[tracer]:
            tracer_HOD = config['HOD_params'][tracer+'_params']
            if tracer == 'LRG':
                Ncent = 0
                Nsat = 0
                for j in range(len(logms)):
                    Ncent += n_cen_LRG(10**logms[j], tracer_HOD['logM_cut'], tracer_HOD['sigma']) * tot_mf[j]
                    Nsat += n_sat_LRG_modified(10**logms[j], 
                    tracer_HOD['logM_cut'], 10**tracer_HOD['logM_cut'], 10**tracer_HOD['logM1'],
                    tracer_HOD['sigma'], tracer_HOD['alpha'], tracer_HOD['kappa']) * tot_mf[j]
                print(tracer, Ncent, Nsat)

# calculate the best fit ngal vs mhalo 
def calc_ngal_bestfit(path2config):

    print("compute expected number of LRGs for a sim, subsampling, no assmbly bias")

    config = yaml.load(open(path2config))

    sim_params = config['sim_params']
    fit_params = config['dynesty_fit_params']    
    clustering_params = config['clustering_params']
    HOD_params = config['HOD_params']
    data_params = config['data_params']
    simname = sim_params['sim_name'] # "AbacusSummit_base_c000_ph006"
    simdir = sim_params['sim_dir']
    z_mock = sim_params['z_mock']
    cleaned_halos = sim_params['cleaned_halos']
    halo_info_fns = \
    list((Path(simdir) / Path(simname) / 'halos' / ('z%4.3f'%z_mock) / 'halo_info').glob('*.asdf'))
    numslabs = len(halo_info_fns)

    newData = xirppi_Data(data_params, HOD_params)
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

    # compute halo mass function
    log_mbins = np.linspace(12, 15.589051329048312, 201)
    tot_mf = 0
    for i in range(numslabs):
        new_mf, logms = load_slab(i, simdir, simname, z_mock, cleaned_halos, log_mbins)
        tot_mf += new_mf

    # acess best fit hod
    dynesty_config_params = config['dynesty_config_params']
    prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
                                dynesty_config_params['chainsPrefix'])

    datafile = np.load(prefix_chain+'_results.npz', allow_pickle = True)
    res1 = datafile['res'].item()

    # print the max logl fit
    logls = res1['logl']
    indmax = np.argmax(logls)
    hod_params = res1['samples'][indmax]
    print("best fit", hod_params, logls[indmax])

    # determine ic
    for key in fit_params.keys():
        tracer_type = fit_params[key][-1]
        mapping_idx = fit_params[key][0]
        #tracer_type = param_tracer[params[mapping_idx, -1]]
        # newBall.tracers[tracer_type][key] = hod_params[mapping_idx]
        if key == 'sigma':
            newBall.tracers[tracer_type][key] = 10**hod_params[mapping_idx]
        else:
            newBall.tracers[tracer_type][key] = hod_params[mapping_idx] 

        if key == 'Acent' or key == 'Asat':
            raise RuntimeWarning("assembly bias turned on, not suitable for this calculation")
        if key == 'Bcent' or key == 'Bsat':
            raise RuntimeWarning("environmental bias turned on, not suitable for this calculation")

    newBall.tracers['LRG']['ic'] = 1
    ngal_dict = newBall.compute_ngal(Nthread = 32)[0]
    N_lrg = ngal_dict['LRG']
    newic = min(1, newData.num_dens_mean['LRG']*newBall.params['Lbox']**3/N_lrg)
    print("ic ", newic)
    print(newBall.tracers['LRG'])

    # calculate number of centrals and satellites
    Ncents = np.zeros(len(logms))
    Nsats = np.zeros(len(logms))
    for j in range(len(logms)):
        Ncents[j] = n_cen_LRG(10**logms[j], newBall.tracers['LRG']['logM_cut'], newBall.tracers['LRG']['sigma']) * tot_mf[j] * newic
        Nsats[j] = n_sat_LRG_modified(10**logms[j], 
        newBall.tracers['LRG']['logM_cut'], 10**newBall.tracers['LRG']['logM_cut'], 10**newBall.tracers['LRG']['logM1'],
        newBall.tracers['LRG']['sigma'], newBall.tracers['LRG']['alpha'], newBall.tracers['LRG']['kappa']) * tot_mf[j] * newic
    np.savez("./dynesty/plot_data/data_Ngal_"+dynesty_config_params['chainsPrefix'], 
        logms = logms, ncents = Ncents, nsats = Nsats)

    # compute typical mass per galaxy
    typical_mass = np.sum((Ncents + Nsats)*10**logms) / np.sum(Ncents + Nsats)
    print("typical mass per galaxy ", np.log10(typical_mass), np.sum(Nsats)/np.sum(Ncents+Nsats))
    print(Ncents+Nsats, logms)

# calculate the best fit ngal vs mhalo 
def calc_Mavg_bestfit(path2config, load_bestfit = True):

    print("compute expected number of LRGs for a sim, subsampling, no assmbly bias")

    config = yaml.load(open(path2config))

    sim_params = config['sim_params']
    fit_params = config['dynesty_fit_params']    
    clustering_params = config['clustering_params']
    HOD_params = config['HOD_params']
    data_params = config['data_params']
    simname = sim_params['sim_name'] # "AbacusSummit_base_c000_ph006"
    simdir = sim_params['sim_dir']
    z_mock = sim_params['z_mock']
    cleaned_halos = sim_params['cleaned_halos']
    halo_info_fns = \
    list((Path(simdir) / Path(simname) / 'halos' / ('z%4.3f'%z_mock) / 'halo_info').glob('*.asdf'))
    numslabs = len(halo_info_fns)

    newData = xirppi_Data(data_params, HOD_params)
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

    # # compute halo mass function
    # log_mbins = np.linspace(12, 15.589051329048312, 201)
    # tot_mf = 0
    # for i in range(numslabs):
    #     new_mf, logms = load_slab(i, simdir, simname, z_mock, cleaned_halos, log_mbins)
    #     tot_mf += new_mf

    # print("mf", tot_mf)
    if load_bestfit:
        # acess best fit hod
        dynesty_config_params = config['dynesty_config_params']
        prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
                                    dynesty_config_params['chainsPrefix'])

        datafile = np.load(prefix_chain+'_results.npz', allow_pickle = True)
        res1 = datafile['res'].item()

        # print the max logl fit
        logls = res1['logl']
        indmax = np.argmax(logls)
        hod_params = res1['samples'][indmax]
        print("best fit", hod_params, logls[indmax])

        # hod_params = [ 1.28350351e+01,  1.40692269e+01, -2.79634291e+00,  8.89787715e-01,
        #   9.55370398e-01,  1.74691015e-01,  1.03191577e+00, -1.26728445e-02, -3.74249020e-01]
        # hod_params = [12.80287797, 13.99974778, -2.93816538,  1.03357034,  0.3651158,   0.17626678,
        # 1.00171066, -0.04554104, -0.16908941]
        # determine ic
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
    

    mock_dict = newBall.run_hod(newBall.tracers, newBall.want_rsd, Nthread = 64)
    print(newBall.tracers['LRG'])
    print("average halo mass ", np.log10(np.mean(mock_dict['LRG']['mass'])))
    Ncent = mock_dict['LRG']['Ncent']
    Ntot = len(mock_dict['LRG']['x'])
    print("ic ", newBall.tracers['LRG']['ic'])
    print('satellite fraction ', (Ntot - Ncent)/Ntot)

def plot_ngal():
    # file_uncleaned = np.load("./dynesty/plot_data/data_Ngal_wp_base_uncleaned_skip3bins.npz")
    # file_cleaned = np.load("./dynesty/plot_data/data_Ngal_wp_base_cleaned_skip3bins.npz")
    
    # file_uncleaned = np.load("./dynesty/plot_data/data_Ngal_xi_base_velbias_nlive500_uncleaned_logsigma_skipcol.npz")
    file_cleaned = np.load("./dynesty/plot_data/data_Ngal_xi_base_velbias_nlive500_cleaned_logsigma_skipcol.npz")
    
    fig, ax = pl.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [3, 1]}, figsize = (6,6))
    # ax[0].plot(file_uncleaned['logms'], file_uncleaned['ncents'], 'r--', label = 'uncleaned')
    # ax[0].plot(file_uncleaned['logms'], file_uncleaned['nsats'], 'b--')
    ax[0].plot(file_cleaned['logms'], file_cleaned['ncents'], 'r-', label = 'cleaned')
    ax[0].plot(file_cleaned['logms'], file_cleaned['nsats'], 'b-')
    ax[0].set_yscale('log')
    ax[0].set_xlim(12.6, 15.1)
    ax[0].set_ylim(5e-1, 5e5)
    ax[0].set_ylabel('$N_{\mathrm{gal}}$')
    ax[0].legend(loc='best')

    # ax[1].plot(file_uncleaned['logms'], (file_cleaned['ncents'] - file_uncleaned['ncents'])/file_uncleaned['ncents'],
     # 'r', alpha = 0.8, label = 'central')
    # ax[1].plot(file_uncleaned['logms'], (file_cleaned['nsats'] - file_uncleaned['nsats'])/file_uncleaned['nsats'],
    #  'b', alpha = 0.8, label = 'satellite')
    ax[1].set_ylim(-9, 9)
    ax[1].axhline(y = 0, xmin = 0, xmax = 1, ls = ':')
    ax[1].set_ylabel('$\delta N_{\mathrm{gal}}$')
    ax[1].set_xlabel('$\log M_h$')
    ax[1].legend(loc='best')

    pl.tight_layout()
    fig.savefig("./plots/plot_ngal_xi_base_velbias_cleaning_skipcol.pdf", dpi = 200)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())
    # calc_ngal_bestfit(**args)
    calc_Mavg_bestfit(**args)
    # halo_mfunc(**args)
    # plot_mfunc(**args)
    # plot_ngal()


