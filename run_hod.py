#!/usr/bin/env python3
'''
This is a script for generating HOD mock catalogs.

Usage
-----
$ python ./run_hod.py --help
'''

import os
import glob
import time

import yaml
import numpy as np
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import gridspec
from matplotlib import rc, rcParams
rcParams.update({'font.size': 13})

from abacusnbody.hod.abacus_hod import AbacusHOD
from likelihood_boss import xirppi_Data, wp_Data

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/boss_lrg_xi.yaml'

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


def main(path2config, params = None, load_bestfit = True):

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
    newData = xirppi_Data(data_params, HOD_params)
    
    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    write_to_disk = HOD_params['write_to_disk']
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1)
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']
    
    # create a new abacushod object
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    # # print("mf", tot_mf)
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
    print(N_lrg)
    newBall.tracers['LRG']['ic'] = min(1, newData.num_dens_mean['LRG']*newBall.params['Lbox']**3/N_lrg)
    print(newBall.tracers)
    mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk = False, Nthread = 64)
    # print("average halo mass ", np.log10(np.mean(mock_dict['LRG']['mass'])))
    # Ncent = mock_dict['LRG']['Ncent']
    # Ntot = len(mock_dict['LRG']['x'])
    # print("ic ", newBall.tracers['LRG']['ic'])
    # print('satellite fraction ', (Ntot - Ncent)/Ntot)

    # wps = newBall.compute_wp(mock_dict, newBall.rpbins, newBall.pimax, newBall.pi_bin_size, Nthread = 16)
    # xis = newBall.compute_xirppi(mock_dict, newBall.rpbins, newBall.pimax, newBall.pi_bin_size, Nthread = 16)
    # print(wps['LRG_LRG'])
    # np.savez("./data/data_wp_base", wp = wps['LRG_LRG'], xi = xis['LRG_LRG'])

    # clustering = newBall.compute_multipole(mock_dict, newBall.rpbins, newBall.pimax, Nthread = 16)
    compare_large_scale(newBall, mock_dict)
    # print(clustering)


def compare_large_scale(newBall, mock_dict):
    rbins = np.linspace(0, 200, 51)
    rmids = 0.5*(rbins[1:] + rbins[:-1])
    xi0 = np.loadtxt("./hong_cmass_large_scale/xi0_cmass_linbins_sys_nj400_z0.46-0.60")[:,1]
    xi0_err = np.loadtxt("./hong_cmass_large_scale/xi0_cmass_linbins_sys_nj400_z0.46-0.60")[:,2]
    xi2 = np.loadtxt("./hong_cmass_large_scale/xi2_cmass_linbins_sys_nj400_z0.46-0.60")[:,1]
    xi2_err = np.loadtxt("./hong_cmass_large_scale/xi2_cmass_linbins_sys_nj400_z0.46-0.60")[:,2]
    xi4 = np.loadtxt("./hong_cmass_large_scale/xi4_cmass_linbins_sys_nj400_z0.46-0.60")[:,1]
    xi4_err = np.loadtxt("./hong_cmass_large_scale/xi4_cmass_linbins_sys_nj400_z0.46-0.60")[:,2]

    model_multipoles = newBall.compute_multipole(mock_dict, rbins, 200, Nthread = 8)['LRG_LRG'][len(rmids):]

    fig = pl.figure(figsize = (6, 5.4))
    imin = 1
    imax = 35
    pl.errorbar(rmids[imin:imax], rmids[imin:imax]**2*xi0[imin:imax], yerr = rmids[imin:imax]**2*xi0_err[imin:imax], 
        c = 'b', ls = '--', marker = '.', label = 'CMASS', alpha = 0.8)
    pl.errorbar(rmids[imin:imax]+0.5, rmids[imin:imax]**2*xi2[imin:imax], yerr = rmids[imin:imax]**2*xi2_err[imin:imax], 
        c = 'r', ls = '--', marker = '.', alpha = 0.8)
    pl.errorbar(rmids[imin:imax]+1, rmids[imin:imax]**2*xi4[imin:imax], yerr = rmids[imin:imax]**2*xi4_err[imin:imax], 
        c = 'g', ls = '--', marker = '.', alpha = 0.8)

    pl.plot(rmids[imin:imax], rmids[imin:imax]**2*model_multipoles[:len(rmids)][imin:imax], c = 'b', 
        ls = '-', marker = '.', label = '$l = 0$', alpha = 0.8)
    pl.plot(rmids[imin:imax]+0.5, rmids[imin:imax]**2*model_multipoles[len(rmids):2*len(rmids)][imin:imax], c = 'r', 
        ls = '-', marker = '.', label = '$l = 2$', alpha = 0.8)
    pl.plot(rmids[imin:imax]+1, rmids[imin:imax]**2*model_multipoles[2*len(rmids):3*len(rmids)][imin:imax], c = 'g', 
        ls = '-', marker = '.', label = '$l = 4$', alpha = 0.8)
    pl.xlabel('$r$ (Mpc/$h$)')
    # pl.xscale('log')
    pl.ylabel('$r^2\\xi_l(r)$ (Mpc$^2$/$h^2$)')

    pl.legend(loc='best', fontsize = 10)
    pl.tight_layout()
    fig.savefig("./plots/plot_compare_large_scale_sys.pdf", dpi = 200)


def plot_A_signature(path2config):
    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    fit_params = config['dynesty_fit_params']    
    newData = xirppi_Data(data_params, HOD_params)
    
    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    write_to_disk = HOD_params['write_to_disk']
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1)

    rs = 0.5*(rpbins[1:] + rpbins[:-1])
    Acentp = np.load("./data/data_wp_Acent+0.2.npz")
    Acentn = np.load("./data/data_wp_Acent-0.2.npz")
    Asatp = np.load("./data/data_wp_Asat+0.2.npz")
    Asatn = np.load("./data/data_wp_Asat-0.2.npz")
    Bcentp = np.load("./data/data_wp_Bcent+0.2.npz")
    Bcentn = np.load("./data/data_wp_Bcent-0.2.npz")
    Bsatp = np.load("./data/data_wp_Bsat+0.2.npz")
    Bsatn = np.load("./data/data_wp_Bsat-0.2.npz")
    base = np.load("./data/data_wp_base.npz")
    myfiles = [Acentp, Acentn, Asatp, Asatn, Bcentp, Bcentn, Bsatp, Bsatn]
    pnames = ['$A_\mathrm{cent}$', '$A_\mathrm{sat}$', '$B_\mathrm{cent}$', '$B_\mathrm{sat}$']

    # make the difference plots
    fig = pl.figure(figsize=(11, 16))
    gs = gridspec.GridSpec(4, 3, width_ratios = [20, 20, 1]) 
    counter = 0
    pim = 30
    mycmap2 = cm.get_cmap('bwr')
    for i in range(4):
        ax1 = fig.add_subplot(gs[counter])
        ax1.set_xlabel('$r_{\perp}$ ($h^{-1} \mathrm{Mpc}$)')
        # ax1.set_ylabel('$r_{\perp} \Delta w_p$ ($h^{-1} \mathrm{Mpc})^2$')
        ax1.set_ylabel('$\delta w_p$')
        ax1.axhline(y = 0, xmin = 0, xmax = 1, c = 'k', ls = '--', alpha = 0.8)
        ax1.plot(rs, myfiles[i*2]['wp'] / base['wp'] - 1, c = 'r', label = "$\Delta$" + pnames[i] + " = 0.2")
        ax1.plot(rs, myfiles[i*2+1]['wp'] / base['wp'] - 1, c = 'b', label = "$\Delta$" + pnames[i] + " = -0.2")
        ax1.set_xscale('log')
        ax1.set_xlim(0.15, 30)
        if i == 2:
            ax1.set_ylim(-0.3, 0.3)
        else:
            ax1.set_ylim(-0.04, 0.04)
        ax1.legend(loc='best', prop={'size': 13})

        delta_xi_norm = (myfiles[i*2]['xi'] - myfiles[i*2+1]['xi']) / base['xi']
        ax2 = fig.add_subplot(gs[counter + 1])
        if i == 2:
            vm = 0.5
        else:
            vm = 0.15
        col2 = ax2.imshow(delta_xi_norm.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
            extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, 30], 
            cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=-vm, vmax=vm))
        # ax2.set_xscale('log')
        ax2.set_xlabel('$\log r_\perp$ ($h^{-1}$Mpc)')
        ax2.set_ylabel('$\pi$ ($h^{-1}$Mpc)')

        ax3 = fig.add_subplot(gs[counter + 2])
        cbar = pl.colorbar(col2, cax = ax3)
        cbar.set_label("$\\xi_+/\\xi_- - 1$", labelpad = 10)
        # cbar.ax.xaxis.set_label_position('top')
        # cbar.set_ticks(np.linspace(-1, 1, num = 5))

        # cbar.set_label('$(\\xi_{\\rm{mock}}-\\xi_{\\rm{BOSS}})/\sigma(\\xi)$', rotation = 270, labelpad = 20)

        counter += 3
    pl.tight_layout()
    plotname = "./plots/plot_deriv_AB"
    fig.savefig(plotname+".pdf", dpi = 300)

    fig = pl.figure(figsize = (4.5, 4))
    pl.plot(rs, rs*(Acentp['wp'] - Acentn['wp'])/0.4, 'o-C0', label = '$x = A_\mathrm{cent}$')
    pl.plot(rs, rs*(Asatp['wp'] - Asatn['wp'])/0.4, '^-C1', label = '$x = A_\mathrm{sat}$')
    pl.axhline(y = 0, xmin = 0, xmax = 1, c = 'k', ls = '--', alpha = 0.8)
    pl.xlabel('$r_p$ ($h^{-1}$Mpc)')
    pl.xscale('log')
    pl.ylabel('$r_p d w_p/dx$')
    pl.legend(loc = 'best')
    pl.tight_layout()
    fig.savefig("./plots/plot_deriv_A.pdf", dpi = 200)

    fig = pl.figure(figsize = (4.9, 4))
    pl.plot(rs, rs*(Bcentp['wp'] - Bcentn['wp'])/0.4, 'o-C0', label = '$x = B_\mathrm{cent}$')
    pl.plot(rs, rs*(Bsatp['wp'] - Bsatn['wp'])/0.4, '^-C1', label = '$x = B_\mathrm{sat}$')
    pl.axhline(y = 0, xmin = 0, xmax = 1, c = 'k', ls = '--', alpha = 0.8)
    pl.xlabel('$r_p$ ($h^{-1}$Mpc)')
    pl.xscale('log')
    pl.ylabel('$r_p d w_p/dx$')
    pl.legend(loc = 'best')
    pl.tight_layout()
    fig.savefig("./plots/plot_deriv_B.pdf", dpi = 200)


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":


    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())

    main(**args)
    # plot_A_signature(**args)

    # # run a series of simulations
    # param_dict = {
    # 'sim_params' :
    #     {
    #     'sim_name': 'AbacusSummit_base_c000_ph006',                                 # which simulation 
    #     # 'sim_dir': '/mnt/gosling2/bigsims/',                                        # where is the simulation
    #     'sim_dir': '/mnt/marvin1/syuan/scratch/bigsims/',                                        # where is the simulation
    #     'output_dir': '/mnt/marvin1/syuan/scratch/data_mocks_georgios',          # where to output galaxy mocks
    #     'subsample_dir': '/mnt/marvin1/syuan/scratch/data_summit/',                 # where to output subsample data
    #     'z_mock': 0.5,                                                             # which redshift slice
    #     'Nthread_load': 7,                                                          # number of thread for organizing simulation outputs (prepare_sim)
    #     'cleaned_halos': False
    #     }
    # }
    # # for i in range(25):
    # #     param_dict['sim_params']['sim_name'] = 'AbacusSummit_base_c000_ph'+str(i).zfill(3)
    # #     main(**args, params = param_dict)

    # other_cosmologies = [
    # # 'AbacusSummit_base_c100_ph000',
    # # 'AbacusSummit_base_c101_ph000',
    # # 'AbacusSummit_base_c102_ph000',
    # # 'AbacusSummit_base_c103_ph000',
    # # 'AbacusSummit_base_c112_ph000',
    # # 'AbacusSummit_base_c113_ph000',
    # # 'AbacusSummit_base_c104_ph000',
    # # 'AbacusSummit_base_c105_ph000',
    # # 'AbacusSummit_base_c109_ph000'
    # 'AbacusSummit_base_c108_ph000',
    # 'AbacusSummit_base_c009_ph000'
    # ]
 
    # for ecosmo in other_cosmologies:
    #     print(ecosmo)
    #     param_dict['sim_params']['sim_name'] = ecosmo
    #     main(**args, params = param_dict)
