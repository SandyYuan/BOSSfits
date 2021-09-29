import numpy as np 
import time
import scipy.integrate as integrate
from scipy import interpolate
import yaml
import argparse
from astropy.io import ascii

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import rc, rcParams
rcParams.update({'font.size': 13})

from likelihood_boss import sigma_Data
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

# a function that outputs the baryonic correction to the lensing measurement given a set of gNFW params

def gNFW(R, R200c, params):
    x = R/R200c
    rho0 = params['rho0']
    xck = params['xck']
    gammak = params['gammak']
    alphak = params['alphak']
    betak = params['betak']
    rhoc = params['rhoc']
    fb = params['fb']
    return rhoc * fb * rho0 * (x/xck)**gammak * (1 + (x/xck)**alphak)**(-(betak - gammak)/alphak)

def sigma_baryon(R, R200c, params):
    start = time.time()
    result = integrate.quad(lambda x: 2*gNFW(np.sqrt(R**2 + x**2), R200c, params), 0, 100)
    return result[0]

def delta_sigma_baryon(rbins, R200c, params):
    # higher resolution than rs
    rbins_fine = np.logspace(-6, np.log10(rbins[-1]), 501)
    rs_fine = 0.5*(rbins_fine[1:] + rbins_fine[:-1])

    sigmas = np.zeros(len(rs_fine))
    for i in range(len(rs_fine)):
        sigmas[i] = sigma_baryon(rs_fine[i], R200c, params)

    mass_cum = np.zeros(len(rbins_fine))
    mass_cum[0] = sigma_baryon(0.5*rbins_fine[0], R200c, params) * np.pi * rbins_fine[0]**2
    for i in range(1, len(rbins_fine)):
        mass_cum[i] = mass_cum[i-1] + sigmas[i-1]*np.pi*(rbins_fine[i]**2 - rbins_fine[i-1]**2)

    sigmas_cum = mass_cum / (np.pi*rbins_fine**2)
    delta_sigma_fine = sigmas_cum[:-1] - sigmas # first bin invalid
    delta_sigma_interp = interpolate.interp1d(rs_fine, delta_sigma_fine)

    rs_coarse = 10**(0.5*(np.log10(rbins[1:]) + np.log10(rbins[:-1])))
    return delta_sigma_interp(rs_coarse)

# def estimate_R200c(params):
#     fsummit = np.load("./data/data_summit_particle_rs_stack_13.43_13.45.npz")
#     rs_summit = fsummit['rs']
#     nhalos_summit = fsummit['nhalos']
#     mpart_summit = fsummit['mpart']
#     rbins = np.linspace(0, 2, 301)
#     rs = 0.5*(rbins[1:] + rbins[:-1])
#     delta_rs = rbins[1:] - rbins[:-1]
#     mass_in_shell = np.histogram(rs_summit, bins = rbins)[0] * mpart_summit / 0.03 / nhalos_summit #  / (4 * np.pi * rs**2) / delta_rs
#     mass_in_ball = np.cumsum(mass_in_shell)
#     density_so = mass_in_ball / (4/3*np.pi*rbins[1:]**3)

#     return rs[np.argmin(abs(density_so - 200*params['rhoc']))]

def compile_halo_id_r95(simname = "/AbacusSummit_base_c000_ph000"):
    ids = []
    r95s = []
    for i in range(34):
        print(i)
        cat = CompaSOHaloCatalog(
            '/mnt/alan1/sbose/scratch/data/cleaned_catalogues'+simname+'/halos/z0.500/halo_info/halo_info_'\
            +str(i).zfill(3)+'.asdf', fields = ['id', 'r95_L2com'], cleaned_halos = True)
        halos = cat.halos
        ids += [halos['id']]
        r95s += [halos['r95_L2com']]
    ids = np.concatenate(ids)
    r95s = np.concatenate(r95s)
    np.savez("./data/halos_id_r95", id = ids, r95 = r95s)

#for the 13.43 - 13.45 sample, r200c = 0.44, r95 = 0.58
def R200c_weights(params):
    # function that loads a mock catalog and computes R200cs with weights
    path2mock = "/mnt/marvin1/syuan/scratch/data_mocks_summit_new/AbacusSummit_base_c000_ph000/z0.500/rsdfit_base_B_LRGs.dat"
    mocks = ascii.read(path2mock)
    gal_ids = mocks['id']

    halos = np.load("./data/halos_id_r95.npz")
    halo_ids = halos['id']
    halo_r95s = halos['r95']

    sorted_indices = halo_ids.argsort()
    halo_ids = halo_ids[sorted_indices]
    halo_r95s = halo_r95s[sorted_indices]
    r95_indices = np.searchsorted(np.array(halo_ids), np.array(gal_ids))
    r95s_gal = halo_r95s[r95_indices]

    r95s_dist, rbins = np.histogram(r95s_gal, bins = 100)
    r95s_dist_norm = r95s_dist / np.sum(r95s_dist)
    np.savez("./data/gal_r95_dist", dist = r95s_dist_norm, bins = rbins)


def main(path2config, combo_key = 'LRG_LRG'):
    params = {}
    params['rho0'] = 10**2.8
    params['xck'] = 0.6
    params['gammak'] = -0.2
    params['alphak'] = 1
    params['betak'] = 2.6

    # cosmology params
    h = 0.6736
    H = h*100 * 1000 / 3.086e22 # s-1
    params['rhoc'] = 3*H**2/(8*np.pi*6.67e-11) / h**2 / 2e30 * (3.086e22)**3 # h2 Msun Mpc-3
    params['fb'] = 0.02237 / (0.12 + 0.02237)

    config = yaml.load(open(path2config))
    sigma_params = config['sigma_params']
    HOD_params = config['HOD_params']
    data_params = config['data_params']
    newData = sigma_Data(data_params, HOD_params)

    rbins = np.array(sigma_params['rbins']) 
    rs = 0.5*(rbins[1:] + rbins[:-1])

    R200c_weights(params)
    # load up all the possible r200cs, compute their sigma_b, take weighted average
    r95_dist = np.load("./data/gal_r95_dist.npz")
    r200_dist = r95_dist['dist']
    r200_mids = 0.5*(r95_dist['bins'][1:] + r95_dist['bins'][:-1]) / 0.58 * 0.44 # approximate conversion

    sigma_baryon_per_r = np.zeros((len(r200_mids), len(rs)))
    for i in range(len(r200_mids)):
        start = time.time()
        sigma_baryon_per_r[i] = delta_sigma_baryon(rbins, r200_mids[i], params)
        print(i, time.time() - start)
    sigma_baryon = np.sum(sigma_baryon_per_r * r200_dist[:, None], axis = 0) / np.sum(r200_dist)

    data_x = newData.rs[combo_key]
    data_y = newData.rs[combo_key] * newData.deltasigma[combo_key]
    data_yerr = newData.rs[combo_key] * np.sqrt(np.diag(newData.covs[combo_key]))

    fig, ax = pl.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [3, 1]}, figsize = (6,6))
    ax[0].plot(data_x, data_y - data_x*sigma_baryon/1e12, label = 'no baryons')
    ax[0].plot(data_x, (data_y - data_x*sigma_baryon/1e12)/(1-params['fb']), ls='--')
    ax[0].errorbar(data_x, data_y, yerr = data_yerr, label = 'observed')
    # ax[0].set_xlabel('$r_p$ ($h^{-1}$ Mpc)')
    ax[0].set_ylabel('$r_p \Delta \Sigma$ (Mpc $M_{\odot}$ pc$^{-2}$)')
    ax[0].set_xscale('log')
    ax[0].legend(loc='best')

    ax[1].plot(data_x, (data_x*sigma_baryon/1e12)/data_y)
    ax[1].fill_between(data_x, data_yerr/data_y, -data_yerr/data_y, alpha = 0.4, color = 'C1')
    ax[1].axhline(y = 0, xmin = 0, xmax = 1, ls = ':', color = 'k')
    ax[1].set_xlabel('$r_p$ ($h^{-1}$ Mpc)')
    ax[1].set_ylabel('$\delta \Sigma$')

    pl.tight_layout()
    pl.savefig('./plots_lensing/plot_lensing_baryons.pdf', dpi = 200)


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    DEFAULTS = {}
    DEFAULTS['path2config'] = 'config/boss_lrg_xi.yaml'

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    

    # sigma_baryon(1, 0.8, params)
    main(**args)