
import os
from pathlib import Path
import yaml

import numpy as np
import random
import time
from astropy.table import Table
import h5py
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator
from itertools import repeat
import argparse

from abacusnbody.data import read_abacus
from glob import glob
import astropy.table
from scipy import spatial

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

import multiprocessing
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import rc, rcParams
rcParams.update({'font.size': 11})

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/abacus_hod.yaml'

def load_slab(i, savedir, simdir, simname, z_mock, cleaning):
    print(i)
    slabname = simdir+simname+'/halos/z'+str(z_mock).ljust(5, '0')\
    +'/halo_info/halo_info_'+str(i).zfill(3)+'.asdf'

    cat = CompaSOHaloCatalog(slabname, subsamples=dict(A=True, rv=True), 
        fields = ['N', 'x_L2com', 'npstartA', 'npoutA'], cleaned_halos = cleaning)
    halos = cat.halos
    if cleaning:
        halos = halos[halos['N'] > 0]

    parts = cat.subsamples
    header = cat.header
    Lbox = cat.header['BoxSizeHMpc']
    Mpart = header['ParticleMassHMsun'] # msun / h 
    H0 = header['H0']
    h = H0/100.0

    mask_halos = (halos['N']*Mpart > 10**(13.44)) & (halos['N']*Mpart < 10**(13.46))
    maskedhalos = halos[mask_halos]
    maskedhalos_pstart = maskedhalos['npstartA']
    maskedhalos_pnum = maskedhalos['npoutA']
    all_delta_pos = []
    for j in np.arange(len(maskedhalos)):
        newparts = parts[maskedhalos_pstart[j]: maskedhalos_pstart[j] + maskedhalos_pnum[j]]
        delta_pos = newparts['pos'] - maskedhalos['x_L2com'][j]
        all_delta_pos += [delta_pos]
    all_delta_pos = np.vstack(all_delta_pos)
    return all_delta_pos, len(maskedhalos), Mpart

def load_allparts(simdir, simname, z_mock, downsample = 1):
    allp = []
    for fn in glob(simdir+simname+'/halos/z'+str(z_mock).ljust(5, '0')+'/*_rv_*/*.asdf'):
        print(fn)
        newparts = read_abacus.read_asdf(fn, load_vel=False)
        if downsample < 1:
            sub_indices = np.random.choice(np.arange(len(newparts)), int(len(newparts)*downsample), replace = False)
            newparts = newparts[sub_indices]
        allp += [newparts]
    # concatenate into one big table
    allp = np.array(astropy.table.vstack(allp)['pos'])
    return allp 

def load_slab_allparts(i, savedir, simdir, simname, z_mock, cleaning, tree, allp):
    print(i)
    slabname = simdir+simname+'/halos/z'+str(z_mock).ljust(5, '0')\
    +'/halo_info/halo_info_'+str(i).zfill(3)+'.asdf'

    cat = CompaSOHaloCatalog(slabname, fields = ['N', 'x_L2com'], cleaned_halos = cleaning)
    halos = cat.halos
    if cleaning:
        halos = halos[halos['N'] > 0]

    header = cat.header
    Lbox = cat.header['BoxSizeHMpc']
    Mpart = header['ParticleMassHMsun'] # msun / h 
    H0 = header['H0']
    h = H0/100.0

    mask_halos = (halos['N']*Mpart > 10**(13.43)) & (halos['N']*Mpart < 10**(13.45))
    maskedhalos = halos[mask_halos]

    all_delta_pos = []
    for j in np.arange(len(maskedhalos)):
        halo_p = maskedhalos['x_L2com'][j]
        newindices = tree.query_ball_point(halo_p, r = 10**(1.5))
        print(j, len(newindices))

        all_delta_pos += [allp[newindices] - halo_p]
    all_delta_pos = np.vstack(all_delta_pos)
    return all_delta_pos, len(maskedhalos), Mpart

def main(path2config):
    config = yaml.load(open(path2config))
    simname = config['sim_params']['sim_name'] # "AbacusSummit_base_c000_ph006"
    simdir = config['sim_params']['sim_dir']
    z_mock = config['sim_params']['z_mock']
    savedir = config['sim_params']['subsample_dir']+simname+"/z"+str(z_mock).ljust(5, '0') 
    cleaning = config['sim_params']['cleaned_halos']
    halo_info_fns = \
    list((Path(simdir) / Path(simname) / 'halos' / ('z%4.3f'%z_mock) / 'halo_info').glob('*.asdf'))
    numslabs = len(halo_info_fns)

    # full_delta_pos = []
    # numhalos = 0
    # for i in range(numslabs):
    #     newresults = load_slab(i, savedir, simdir, simname, z_mock, cleaning)
    #     full_delta_pos += [newresults[0]]
    #     numhalos += newresults[1]
    # full_delta_pos = np.vstack(full_delta_pos)

    # use full particle subsample
    downsampling = 0.005
    # allps = load_allparts(simdir, simname, z_mock, downsample = downsampling)
    subsample_name = Path(savedir) / ('parts_all_down_%4.3f.h5'%downsampling)
    newfile = h5py.File(subsample_name, 'r')
    full_subsample = np.array(newfile['particles'])
    subsample_tree = spatial.cKDTree(full_subsample)

    full_delta_pos = []
    numhalos = 0
    for i in range(numslabs):
        newresults = load_slab_allparts(i, savedir, simdir, simname, z_mock, cleaning, subsample_tree, full_subsample)
        full_delta_pos += [newresults[0]]
        numhalos += newresults[1]
    full_delta_pos = np.vstack(full_delta_pos)

    rs = np.sqrt(full_delta_pos[:, 0]**2 + full_delta_pos[:, 1]**2 + full_delta_pos[:, 2]**2)

    np.savez("./data/data_summit_allparticle_rs_stack_13.44_13.46", rs = rs, nhalos = numhalos, mpart = newresults[2]/downsampling)

def plot_profile():
    fcosmos = np.load("../s3PCF_fenv/data/data_particle_rs_stack_13.49_13.51.npz")
    rs_cosmos = fcosmos['rs']
    nhalos_cosmos = fcosmos['nhalos']
    mpart_cosmos = fcosmos['mpart']

    fsummit = np.load("./data/data_summit_particle_rs_stack_13.43_13.45.npz")
    rs_summit = fsummit['rs']
    nhalos_summit = fsummit['nhalos']
    mpart_summit = fsummit['mpart']

    fsummitfull = np.load("./data/data_summit_allparticle_rs_stack_13.43_13.45.npz")
    rs_summitfull = fsummitfull['rs']
    nhalos_summitfull = fsummitfull['nhalos']
    mpart_summitfull = fsummitfull['mpart']

    rbins = np.logspace(-2, 0.3, 51)
    rs = 0.5*(rbins[1:] + rbins[:-1])
    delta_rs = rbins[1:] - rbins[:-1]
    rhos_cosmos = np.histogram(rs_cosmos, bins = rbins)[0] * mpart_cosmos * 10 / nhalos_cosmos / (4 * np.pi * rs**2) / delta_rs
    rhos_summit = np.histogram(rs_summit, bins = rbins)[0] * mpart_summit / 0.03 / nhalos_summit / (4 * np.pi * rs**2) / delta_rs
    rhos_summitfull = np.histogram(rs_summitfull, bins = rbins)[0] * mpart_summitfull / 0.03 / nhalos_summitfull / (4 * np.pi * rs**2) / delta_rs

    fig, ax = pl.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [3, 1]}, figsize = (5,6))
    ax[0].set_title('$13.43 < \log M_h < 13.45$')
    ax[0].plot(rs, rs**2*rhos_cosmos, label = 'AbacusCosmos', color = 'C3')
    ax[0].plot(rs, rs**2*rhos_summit, label = 'AbacusSummit', color = 'C0')
    skipbins = 9
    # ax[0].plot(rs[:-skipbins], rs[:-skipbins]**2*rhos_summitfull[:-skipbins], label = 'AbacusSummit full sample', color = 'C0', ls = '--')
    ax[0].legend(loc = 'best')
    ax[0].set_xscale('log')
    # ax[0].set_yscale('log')
    # ax[0].set_xlabel('$r$ (Mpc/h)')
    ax[0].set_ylabel('$r^2 \\rho$ ($M_\odot$ / Mpc)')

    ax[1].plot(rs, (rhos_summit - rhos_cosmos)/rhos_cosmos, color = 'C0')
    # ax[1].plot(rs, (rhos_summitfull - rhos_cosmos)/rhos_cosmos, color = 'C0', ls = '--')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('$r$ (Mpc/h)')
    ax[1].set_ylabel('$\\rho_\mathrm{suumit}/\\rho_\mathrm{cosmos} - 1$')
    ax[1].axhline(y = 0, xmin = 0, xmax = 1, ls = '--', alpha = 0.5)
    # ax[1].set_ylim(-0.6, 0.6)
    pl.tight_layout()
    fig.savefig("./plots/plot_rho_r_13.43-45_matched_new.pdf", dpi = 200)

    # compute 
    delta_cosmos = np.zeros(len(rhos_cosmos)-1)
    delta_summit = np.zeros(len(rhos_summit)-1)
    enclosed_cosmos = 4/3*np.pi*rs[0]**3*rhos_cosmos[0]
    enclosed_summit = 4/3*np.pi*rs[0]**3*rhos_summit[0]   
    for i in range(len(rs)-1):
        delta_cosmos[i] = enclosed_cosmos / (4/3*np.pi*rs[i]**3) - rhos_cosmos[i + 1]
        delta_summit[i] = enclosed_summit / (4/3*np.pi*rs[i]**3) - rhos_summit[i + 1]
        enclosed_cosmos += 4*np.pi*rs[i+1]**2*rhos_cosmos[i + 1]*(rs[i+1] - rs[i])
        enclosed_summit += 4*np.pi*rs[i+1]**2*rhos_summit[i + 1]*(rs[i+1] - rs[i])

    print(delta_cosmos)
    fig, ax = pl.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [3, 1]}, figsize = (5,6))
    ax[0].set_title('$13.43 < \log M_h < 13.45$')
    skipbins = 15
    ax[0].plot(rs[skipbins+1:], rs[skipbins+1:]**2*delta_cosmos[skipbins:], label = 'AbacusCosmos', color = 'C3')
    ax[0].plot(rs[skipbins+1:], rs[skipbins+1:]**2*delta_summit[skipbins:], label = 'AbacusSummit', color = 'C0')
    # ax[0].plot(rs[:-skipbins], rs[:-skipbins]**2*rhos_summitfull[:-skipbins], label = 'AbacusSummit full sample', color = 'C0', ls = '--')
    ax[0].legend(loc = 'best')
    ax[0].set_xscale('log')
    # ax[0].set_yscale('log')
    # ax[0].set_xlabel('$r$ (Mpc/h)')
    ax[0].set_ylabel('$r^2\Delta\\rho$ ($M_\odot$ / Mpc)')

    ax[1].plot(rs[skipbins+1:], (delta_summit[skipbins:] - delta_cosmos[skipbins:])/delta_cosmos[skipbins:], color = 'C0')
    # ax[1].plot(rs, (rhos_summitfull - rhos_cosmos)/rhos_cosmos, color = 'C0', ls = '--')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('$r$ (Mpc/h)')
    ax[1].set_ylabel('$\Delta\\rho_\mathrm{suumit}/\delta\\rho_\mathrm{cosmos} - 1$')
    ax[1].axhline(y = 0, xmin = 0, xmax = 1, ls = '--', alpha = 0.5)
    # ax[1].set_ylim(-0.6, 0.6)
    pl.tight_layout()
    fig.savefig("./plots/plot_deltarho_r_13.43-45_matched_new.pdf", dpi = 200)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())

    # main(**args)
    plot_profile()