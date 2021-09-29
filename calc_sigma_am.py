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
from Corrfunc.theory import wp, xi, DDrppi

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

def calc_halo_part_DD(path2config, mock_key = 'LRG', combo_key = 'LRG_LRG', downsample = 0.005):

    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    dynesty_config_params = config['dynesty_config_params']
    fit_params = config['dynesty_fit_params']    
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    Mpart = newBall.params['Mpart'] # msun / h
    newData = sigma_Data(data_params, HOD_params)

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

    full_subsample = full_subsample % newBall.lbox
    pos_gals = np.load("../summit/data/data_gal_am_ph000_cleaned.npz")['x'] % newBall.lbox

    DD = DDrppi(0, 32, 30, newData.rbins[mock_key+'_'+mock_key], pos_gals[:, 0], pos_gals[:, 1], pos_gals[:, 2], 
        X2=full_subsample[:, 0], Y2=full_subsample[:, 1], Z2=full_subsample[:, 2])

    print(DD['npairs'])
    np.savez("./data/halo_part_DD_ph000_cleaned_"+str(downsample), npairs = DD['npairs'])

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
    Mpart = newBall.params['Mpart'] # msun / h
    newData = sigma_Data(data_params, HOD_params)
    print(newData.rbins[combo_key])

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

    full_subsample = full_subsample % newBall.lbox
    pos_gals = np.load("../summit/data/data_gal_am_ph000_cleaned_cut40000.npz")['x'] % newBall.lbox

    start = time.time()
    delta_sig = mean_delta_sigma(pos_gals, full_subsample, Mpart/0.03/downsample, 
        newData.rbins[combo_key], newBall.lbox)
    print(time.time() - start, delta_sig)
    np.savez("./data_lensing/data_AM_ph000_cleaned_cut40000", rs = newData.rs[combo_key], delta_sig = delta_sig)

def plot_DD(path2config, combo_key = 'LRG_LRG'):
    config = yaml.load(open(path2config))
    HOD_params = config['HOD_params']
    data_params = config['data_params']
    newData = sigma_Data(data_params, HOD_params)
    sim_params = config['sim_params']
    clustering_params = config['clustering_params']
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

    Mpart = newBall.params['Mpart'] # msun / h
    rbins = newData.rbins[combo_key]

    DD_cleaned = np.load("./data/halo_part_DD_ph000_cleaned_0.005.npz")['npairs']
    DD_cleaned_proj = np.sum(DD_cleaned.reshape(len(rbins)-1, 30), axis = 1)
    Mpart_cleaned = Mpart / 0.03 / 0.005
    Ngals_cleaned = 3e-4 * 2000**3

    DD_cosmos = 0
    Ncosmos = 3
    for i in range(Ncosmos):
        DD_cosmos += np.load("../s3PCF_fenv/data/halo_part_DD_"+str(i)+".npz")['npairs']
    DD_cosmos = DD_cosmos / Ncosmos
    DD_cosmos_proj = np.sum(DD_cosmos.reshape(len(rbins)-1, 30), axis = 1)
    Mpart_cosmos = 3.88537e+10 * 200 * 10
    Ngals_cosmos = 3e-4 * 1100**3

    normalization = np.zeros(len(rbins) - 1)
    for i in range(len(rbins) - 1):
        normalization[i] = np.pi*(rbins[i+1]**2 - rbins[i]**2) * 30

    rmids = 0.5*(rbins[1:] + rbins[:-1])

    print(DD_cleaned_proj*Mpart_cleaned/Ngals_cleaned/(DD_cosmos_proj*Mpart_cosmos/Ngals_cosmos) - 1)
    fig = pl.figure(figsize = (4.3, 4))
    pl.xlabel('$r_p$ ($h^{-1}$Mpc)')
    pl.ylabel('$r_p \\rho$ ($M_\odot h$Mpc$^{-2}$)')
    pl.plot(rmids, rmids*DD_cleaned_proj*Mpart_cleaned/Ngals_cleaned/normalization, label = 'summit cleaned')
    pl.plot(rmids, rmids*DD_cosmos_proj*Mpart_cosmos/Ngals_cosmos/normalization, label = 'cosmos')
    pl.xscale('log')
    pl.legend(loc = 'best', fontsize = 7)
    pl.tight_layout()
    fig.savefig("./plots/plot_halo_particle_crosscorr_am.pdf", dpi = 200)



def plot_sigma(path2config, combo_key = 'LRG_LRG'):
    config = yaml.load(open(path2config))
    HOD_params = config['HOD_params']
    data_params = config['data_params']
    newData = sigma_Data(data_params, HOD_params)

    # dsig_uncleaned = np.load("./data_lensing/data_AM_ph000_uncleaned.npz")['delta_sig']
    dsig_cleaned = np.load("./data_lensing/data_AM_ph000_cleaned.npz")['delta_sig']
    print(dsig_cleaned)
    Nsim_cosmos = 10
    dsig_cosmos = 0
    for i in range(Nsim_cosmos):
        dsig_cosmos += np.load("../s3PCF_fenv/data/data_deltasigma_am_"+str(i)+".npz")['delta_sig']
        print(i, np.load("../s3PCF_fenv/data/data_deltasigma_am_"+str(i)+".npz")['delta_sig'])
    dsig_cosmos = dsig_cosmos / Nsim_cosmos

    data_x = newData.rs[combo_key]
    data_y = newData.rs[combo_key] * newData.deltasigma[combo_key]
    data_yerr = newData.rs[combo_key] * np.sqrt(np.diag(newData.covs[combo_key]))

    fig, ax = pl.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [3, 1]}, figsize = (5.2,6))
    # ax[0].plot(data_x, data_x*dsig_uncleaned/1e12, color = 'C3', label = 'uncleaned')
    # ax[0].plot(data_x, data_x*sigma_wp_uncleaned['delta_sig']/1e12, color = 'C3', ls = '--', label = '$w_p$ fit uncleaned')
    ax[0].plot(data_x, data_x*dsig_cleaned/1e12, color = 'C1', label = 'AbacusSummit')
    ax[0].plot(data_x, data_x*dsig_cosmos/1e12, color = 'C2', label = 'AbacusCosmos')
    ax[0].errorbar(data_x, data_y, yerr = data_yerr, label = 'observed', color = 'C0', marker = 'o')
    # ax[0].set_xlabel('$r_p$ ($h^{-1}$ Mpc)')
    ax[0].set_ylabel('$r_p \Delta \Sigma$ (Mpc $M_{\odot}$ pc$^{-2}$)')
    ax[0].set_ylim(2, 12)
    ax[0].set_xscale('log')
    ax[0].legend(loc='best', fontsize = 9)

    # ax[1].plot(data_x, (data_x*dsig_uncleaned/1e12 - data_y)/data_y, color = 'C3')
    # ax[1].plot(data_x, (data_x*sigma_wp_uncleaned['delta_sig']/1e12 - data_y)/data_y, color = 'C3', ls = '--')
    ax[1].plot(data_x, (data_x*dsig_cleaned/1e12 - data_y)/data_y, color = 'C1')
    ax[1].plot(data_x, (data_x*dsig_cosmos/1e12 - data_y)/data_y, color = 'C2')
    ax[1].fill_between(data_x, data_yerr/data_y, -data_yerr/data_y, alpha = 0.4, color = 'C0')
    ax[1].axhline(y = 0, xmin = 0, xmax = 1, ls = ':', color = 'C0')
    ax[1].set_xlabel('$r_p$ ($h^{-1}$ Mpc)')
    ax[1].set_ylabel('$\delta \Sigma$')

    pl.tight_layout()
    pl.savefig('./plots_lensing/plot_lensing_am.pdf', dpi = 200)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    # calc_sigma(**args)
    # calc_halo_part_DD(**args)
    plot_sigma(**args)
    # plot_DD(**args)

