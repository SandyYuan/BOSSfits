"""
class that loads up the sigma per halo and particle data, and computes delta sigma for a given hod
"""

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

from abacusnbody.hod.abacus_hod import AbacusHOD
from abacusnbody.hod.GRAND_HOD import *
from likelihood_boss import sigma_Data

class calc_Sigma(AbacusHOD):
    def __init__(self, abacushod, sigma_params):
        # inheriting variables from abacushod
        self.sim_name = abacushod.sim_name
        self.sim_dir = abacushod.sim_dir
        self.subsample_dir = abacushod.subsample_dir
        self.z_mock = abacushod.z_mock
        self.output_dir = abacushod.output_dir
        self.sigma_params = sigma_params

        self.tracers = abacushod.tracers
        MT = False
        for key in self.tracers.keys():
            if key == 'ELG' or key == 'QSO':
                MT = True

        self.want_ranks = abacushod.want_ranks

        self.halo_data = abacushod.halo_data 
        self.particle_data = abacushod.particle_data

        # loading the sigma data
        savedir = self.subsample_dir+self.sim_name+"/z"+str(self.z_mock).ljust(5, '0') 

        halo_info_fns = \
        list((Path(self.sim_dir) / Path(self.sim_name) / 'halos' / ('z%4.3f'%self.z_mock) / 'halo_info').glob('*.asdf'))
        numslabs = len(halo_info_fns)

        halos_sigma = []
        particles_sigma = []
        for i in range(numslabs):
            filename_halos = savedir+'/halos_xcom_'+str(i)+'_seed600_abacushod_oldfenv'
            filename_particles = savedir+'/particles_xcom_'+str(i)+'_seed600_abacushod_oldfenv'
            if MT:
                filename_halos += '_MT'
                filename_particles += '_MT'
            if self.want_ranks:
                filename_particles += '_withranks'
            new_halos_sigma = np.array(h5py.File(filename_halos+'_new_sigma.h5', 'r')['sigma'])
            new_particles_sigma = np.array(h5py.File(filename_particles+'_new_sigma.h5', 'r')['sigma'])

            halos_sigma += [new_halos_sigma]
            particles_sigma += [new_particles_sigma]

        halos_sigma = np.concatenate(halos_sigma, axis = 0)
        particles_sigma = np.concatenate(particles_sigma, axis = 0)

        self.halo_sigma = halos_sigma
        self.particle_sigma = particles_sigma

    def run_hod(self, tracers = None, Nthread = 16, verbose = False):
        if tracers == None:
            tracers = self.tracers

        mock_sigma = self.gen_sigma(tracers, Nthread, verbose)
        return mock_sigma

    def gen_sigma(self, tracers, Nthread, verbose):
        for tracer in tracers.keys():
            if tracer == 'LRG':
                LRG_HOD = tracers[tracer]
            if tracer == 'ELG':
                ELG_HOD = tracers[tracer]
            if tracer == 'QSO':
                QSO_HOD = tracers[tracer]

        if 'LRG' in tracers.keys():
            want_LRG = True
            # LRG design and decorations
            logM_cut_L, logM1_L, sigma_L, alpha_L, kappa_L = \
                map(LRG_HOD.get, ('logM_cut', 
                                  'logM1', 
                                  'sigma', 
                                  'alpha', 
                                  'kappa'))
            LRG_design_array = np.array([logM_cut_L, logM1_L, sigma_L, alpha_L, kappa_L])

            alpha_c, alpha_s, s, s_v, s_p, s_r, Ac, As, Bc, Bs, ic = \
                map(LRG_HOD.get, ('alpha_c', 
                                'alpha_s',  
                                's', 
                                's_v', 
                                's_p', 
                                's_r',
                                'Acent',
                                'Asat',
                                'Bcent',
                                'Bsat',
                                'ic'))
            LRG_decorations_array = np.array([alpha_c, alpha_s, s, s_v, s_p, s_r, Ac, As, Bc, Bs, ic])
            if ((s != 0) or (s_p != 0) or (s_v != 0) or (s_r != 0)) and (not self.want_ranks):
                raise ValueError("enabling s parameters requires want_ranks flag set to True")
        else:
            # B.H. TODO: this will go when we switch to dictionaried and for loops
            want_LRG = False
            LRG_design_array = np.zeros(5)
            LRG_decorations_array = np.zeros(11)
            
        if 'ELG' in tracers.keys():
            # ELG design
            want_ELG = True
            pmax_E, Q_E, logM_cut_E, kappa_E, sigma_E, logM1_E, alpha_E, gamma_E, A_E = \
                map(ELG_HOD.get, ('p_max',
                                'Q',
                                'logM_cut',
                                'kappa',
                                'sigma',
                                'logM1',
                                'alpha',
                                'gamma',
                                'A_s'))
            ELG_design_array = np.array(
                [pmax_E, Q_E, logM_cut_E, kappa_E, sigma_E, logM1_E, alpha_E, gamma_E, A_E])
            alpha_c_E, alpha_s_E, s_E, s_v_E, s_p_E, s_r_E, Ac_E, As_E, Bc_E, Bs_E = \
                map(ELG_HOD.get, ('alpha_c', 
                                'alpha_s',  
                                's', 
                                's_v', 
                                's_p', 
                                's_r',
                                'Acent',
                                'Asat',
                                'Bcent',
                                'Bsat'))
            ELG_decorations_array = np.array(
                [alpha_c_E, alpha_s_E, s_E, s_v_E, s_p_E, s_r_E, Ac_E, As_E, Bc_E, Bs_E])
            if ((s_E != 0) or (s_p_E != 0) or (s_v_E != 0) or (s_r_E != 0)) and (not self.want_ranks):
                raise ValueError("enabling s parameters requires want_ranks flag set to True")
        else:
            # B.H. TODO: this will go when we switch to dictionaried and for loops
            ELG_design_array = np.zeros(9)
            ELG_decorations_array = np.zeros(10)
            want_ELG = False
            
        if 'QSO' in tracers.keys():
            # QSO design
            want_QSO = True
            pmax_Q, logM_cut_Q, kappa_Q, sigma_Q, logM1_Q, alpha_Q, A_Q = \
                map(QSO_HOD.get, ('p_max',
                                'logM_cut',
                                'kappa',
                                'sigma',
                                'logM1',
                                'alpha',
                                'A_s'))
            QSO_design_array = np.array(
                [pmax_Q, logM_cut_Q, kappa_Q, sigma_Q, logM1_Q, alpha_Q, A_Q])
            alpha_c_Q, alpha_s_Q, s_Q, s_v_Q, s_p_Q, s_r_Q, Ac_Q, As_Q, Bc_Q, Bs_Q = \
                map(QSO_HOD.get, ('alpha_c', 
                                'alpha_s',  
                                's', 
                                's_v', 
                                's_p', 
                                's_r',
                                'Acent',
                                'Asat',
                                'Bcent',
                                'Bsat'))
            QSO_decorations_array = np.array(
                [alpha_c_Q, alpha_s_Q, s_Q, s_v_Q, s_p_Q, s_r_Q, Ac_Q, As_Q, Bc_Q, Bs_Q])
            if ((s_Q != 0) or (s_p_Q != 0) or (s_v_Q != 0) or (s_r_Q != 0)) and (not self.want_ranks):
                raise ValueError("enabling s parameters requires want_ranks flag set to True")
        else:
            # B.H. TODO: this will go when we switch to dictionaried and for loops
            QSO_design_array = np.zeros(7)
            QSO_decorations_array = np.zeros(10)
            want_QSO = False

        start = time.time()
        sum_sigma_L_cent, sum_sigma_E_cent, sum_sigma_Q_cent, nl_cent, ne_cent, nq_cent \
        = calc_Sigma._gen_sigma_halos(self.halo_data['hmass'], self.halo_data['hmultis'], self.halo_data['hrandoms'], 
                self.halo_data['hveldev'], self.halo_data['hdeltac'], self.halo_data['hfenv'], self.halo_sigma,
                LRG_design_array, LRG_decorations_array, ELG_design_array, ELG_decorations_array, QSO_design_array, 
                QSO_decorations_array, want_LRG, want_ELG, want_QSO, Nthread)
        if verbose:
            print("generating mean sigma for centrals took ", time.time() - start)

        start = time.time()
        sum_sigma_L_sats, sum_sigma_E_sats, sum_sigma_Q_sats, nl_sats, ne_sats, nq_sats \
        = calc_Sigma._gen_sigma_particles(self.particle_data['phmass'], self.particle_data['pweights'], self.particle_data['prandoms'], 
                self.particle_data['pdeltac'], self.particle_data['pfenv'], self.want_ranks, 
                self.particle_data['pranks'], self.particle_data['pranksv'], self.particle_data['pranksp'], 
                self.particle_data['pranksr'], self.particle_sigma, 
                LRG_design_array, LRG_decorations_array, ELG_design_array, 
                ELG_decorations_array, QSO_design_array, QSO_decorations_array, 
                want_LRG, want_ELG, want_QSO, Nthread)
        if verbose:
            print("generating satellites took ", time.time() - start)

        mock_sigma = {}
        if want_LRG:
            mock_sigma['LRG'] = (sum_sigma_L_cent + sum_sigma_L_sats) / (nl_cent + nl_sats)
        if want_ELG:
            mock_sigma['ELG'] = (sum_sigma_E_cent + sum_sigma_E_sats) / (ne_cent + ne_sats)
        if want_QSO:
            mock_sigma['QSO'] = (sum_sigma_Q_cent + sum_sigma_Q_sats) / (nq_cent + nq_sats)

        return mock_sigma

    @staticmethod
    @njit(parallel=True, fastmath = True)
    def _gen_sigma_halos(mass, multis, randoms, vdev, deltac, fenv, delsigma, 
        LRG_design_array, LRG_decorations_array, ELG_design_array, 
        ELG_decorations_array, QSO_design_array, QSO_decorations_array, 
        want_LRG, want_ELG, want_QSO, Nthread):

        # parse out the hod parameters 
        logM_cut_L, logM1_L, sigma_L, alpha_L, kappa_L = \
            LRG_design_array[0], LRG_design_array[1], LRG_design_array[2], LRG_design_array[3], LRG_design_array[4]
        ic_L, alpha_c_L, Ac_L, Bc_L = LRG_decorations_array[10], LRG_decorations_array[0], \
            LRG_decorations_array[6], LRG_decorations_array[8]

        pmax_E, Q_E, logM_cut_E, kappa_E, sigma_E, logM1_E, alpha_E, gamma_E, A_E = \
            ELG_design_array[0], ELG_design_array[1], ELG_design_array[2], ELG_design_array[3], ELG_design_array[4],\
            ELG_design_array[5], ELG_design_array[6], ELG_design_array[7], ELG_design_array[8]
        alpha_c_E, Ac_E, Bc_E = ELG_decorations_array[0], ELG_decorations_array[6], ELG_decorations_array[8]

        pmax_Q, logM_cut_Q, kappa_Q, sigma_Q, logM1_Q, alpha_Q, A_Q = \
            QSO_design_array[0], QSO_design_array[1], QSO_design_array[2], QSO_design_array[3], QSO_design_array[4],\
            QSO_design_array[5], QSO_design_array[6]
        alpha_c_Q, Ac_Q, Bc_Q = QSO_decorations_array[0], QSO_decorations_array[6], QSO_decorations_array[8]

        H = len(mass)

        numba.set_num_threads(Nthread)
        hstart = np.rint(np.linspace(0, H, Nthread + 1)) # starting index of each thread

        # keep = np.empty(H, dtype = np.int8) # mask array tracking which halos to keep
        delsigma_L = np.zeros((Nthread, delsigma.shape[-1], 8), dtype = np.float32)
        N_L = np.zeros((Nthread, 8), dtype = np.int32)
        delsigma_E = np.zeros((Nthread, delsigma.shape[-1], 8), dtype = np.float32)
        N_E = np.zeros((Nthread, 8), dtype = np.int32)
        delsigma_Q = np.zeros((Nthread, delsigma.shape[-1], 8), dtype = np.float32)
        N_Q = np.zeros((Nthread, 8), dtype = np.int32)

        # figuring out the number of halos kept for each thread
        for tid in numba.prange(Nthread):
            for i in range(hstart[tid], hstart[tid + 1]):
                # first create the markers between 0 and 1 for different tracers
                LRG_marker = 0
                if want_LRG:
                    # do assembly bias and secondary bias
                    logM_cut_L_temp = logM_cut_L + Ac_L * deltac[i] + Bc_L * fenv[i]
                    LRG_marker += n_cen_LRG(mass[i], logM_cut_L_temp, sigma_L) * ic_L * multis[i]
                ELG_marker = LRG_marker
                if want_ELG:
                    logM_cut_E_temp = logM_cut_E + Ac_E * deltac[i] + Bc_E * fenv[i]
                    ELG_marker += N_cen_ELG_v1(mass[i], pmax_E, Q_E, logM_cut_E_temp, sigma_E, gamma_E) * multis[i]
                QSO_marker = ELG_marker
                if want_QSO:
                    logM_cut_Q_temp = logM_cut_Q + Ac_Q * deltac[i] + Bc_Q * fenv[i]
                    QSO_marker += N_cen_QSO(mass[i], pmax_Q, logM_cut_Q, sigma_Q)

                if randoms[i] <= LRG_marker:
                    delsigma_L[tid, :, 0] += delsigma[i]
                    N_L[tid, 0] += 1
                elif randoms[i] <= ELG_marker:
                    delsigma_E[tid, :, 0] += delsigma[i]
                    N_E[tid, 0] += 1
                elif randoms[i] <= QSO_marker:
                    delsigma_Q[tid, :, 0] += delsigma[i]
                    N_Q[tid, 0] += 1

        delsigma_L = np.sum(delsigma_L, axis = 0)
        delsigma_E = np.sum(delsigma_E, axis = 0)
        delsigma_Q = np.sum(delsigma_Q, axis = 0)
        N_L = np.sum(N_L)
        N_E = np.sum(N_E)
        N_Q = np.sum(N_Q)

        return delsigma_L[:, 0], delsigma_E[:, 0], delsigma_Q[:, 0], N_L, N_E, N_Q


    @staticmethod
    @njit(parallel = True, fastmath = True)
    def _gen_sigma_particles(hmass, weights, randoms, hdeltac, hfenv, enable_ranks,
        ranks, ranksv, ranksp, ranksr, delsigma, 
        LRG_design_array, LRG_decorations_array, ELG_design_array, ELG_decorations_array,
        QSO_design_array, QSO_decorations_array,
        want_LRG, want_ELG, want_QSO, Nthread):

        """
        Generate satellite galaxies in place in memory with a two pass numba parallel implementation. 
        """

        # standard hod design
        logM_cut_L, logM1_L, sigma_L, alpha_L, kappa_L = \
            LRG_design_array[0], LRG_design_array[1], LRG_design_array[2], LRG_design_array[3], LRG_design_array[4]
        alpha_s_L, s_L, s_v_L, s_p_L, s_r_L, Ac_L, As_L, Bc_L, Bs_L, ic_L = \
            LRG_decorations_array[1], LRG_decorations_array[2], LRG_decorations_array[3], LRG_decorations_array[4], \
            LRG_decorations_array[5], LRG_decorations_array[6], LRG_decorations_array[7], LRG_decorations_array[8], \
            LRG_decorations_array[9], LRG_decorations_array[10]

        pmax_E, Q_E, logM_cut_E, kappa_E, sigma_E, logM1_E, alpha_E, gamma_E, A_E = \
            ELG_design_array[0], ELG_design_array[1], ELG_design_array[2], ELG_design_array[3], ELG_design_array[4],\
            ELG_design_array[5], ELG_design_array[6], ELG_design_array[7], ELG_design_array[8]
        alpha_s_E, s_E, s_v_E, s_p_E, s_r_E, Ac_E, As_E, Bc_E, Bs_E = \
            ELG_decorations_array[1], ELG_decorations_array[2], ELG_decorations_array[3], ELG_decorations_array[4], \
            ELG_decorations_array[5], ELG_decorations_array[6], ELG_decorations_array[7], ELG_decorations_array[8], \
            ELG_decorations_array[9]

        pmax_Q, logM_cut_Q, kappa_Q, sigma_Q, logM1_Q, alpha_Q, A_Q = \
            QSO_design_array[0], QSO_design_array[1], QSO_design_array[2], QSO_design_array[3], QSO_design_array[4],\
            QSO_design_array[5], QSO_design_array[6]
        alpha_s_Q, s_Q, s_v_Q, s_p_Q, s_r_Q, Ac_Q, As_Q, Bc_Q, Bs_Q = \
            QSO_decorations_array[1], QSO_decorations_array[2], QSO_decorations_array[3], QSO_decorations_array[4], \
            QSO_decorations_array[5], QSO_decorations_array[6], QSO_decorations_array[7], QSO_decorations_array[8], \
            QSO_decorations_array[9]

        H = len(hmass) # num of particles

        delsigma_L = np.zeros((Nthread, delsigma.shape[-1], 8), dtype = np.float32)
        N_L = np.zeros((Nthread, 8), dtype = np.int32)
        delsigma_E = np.zeros((Nthread, delsigma.shape[-1], 8), dtype = np.float32)
        N_E = np.zeros((Nthread, 8), dtype = np.int32)
        delsigma_Q = np.zeros((Nthread, delsigma.shape[-1], 8), dtype = np.float32)
        N_Q = np.zeros((Nthread, 8), dtype = np.int32)

        numba.set_num_threads(Nthread)
        hstart = np.rint(np.linspace(0, H, Nthread + 1)) # starting index of each thread

        # figuring out the number of particles kept for each thread
        for tid in numba.prange(Nthread): #numba.prange(Nthread):
            for i in range(hstart[tid], hstart[tid + 1]):
                # print(logM1, As, hdeltac[i], Bs, hfenv[i])
                LRG_marker = 0
                if want_LRG:
                    M1_L_temp = 10**(logM1_L + As_L * hdeltac[i] + Bs_L * hfenv[i])
                    logM_cut_L_temp = logM_cut_L + Ac_L * hdeltac[i] + Bc_L * hfenv[i]
                    base_p_L = n_sat_LRG_modified(hmass[i], logM_cut_L_temp, 
                        10**logM_cut_L_temp, M1_L_temp, sigma_L, alpha_L, kappa_L) * weights[i] * ic_L
                    if enable_ranks:
                        decorator_L = 1 + s_L * ranks[i] + s_v_L * ranksv[i] + s_p_L * ranksp[i] + s_r_L * ranksr[i]
                        exp_sat = base_p_L * decorator_L
                    else:
                        exp_sat = base_p_L
                    LRG_marker += exp_sat
                ELG_marker = LRG_marker
                if want_ELG:
                    M1_E_temp = 10**(logM1_E + As_E * hdeltac[i] + Bs_E * hfenv[i])
                    logM_cut_E_temp = logM_cut_E + Ac_E * hdeltac[i] + Bc_E * hfenv[i]
                    base_p_E = N_sat_generic(
                        hmass[i], 10**logM_cut_E_temp, kappa_E, M1_E_temp, alpha_E, A_E) * weights[i]
                    if enable_ranks:
                        decorator_E = 1 + s_E * ranks[i] + s_v_E * ranksv[i] + s_p_E * ranksp[i] + s_r_E * ranksr[i]
                        exp_sat = base_p_E * decorator_E
                    else:
                        exp_sat = base_p_E
                    ELG_marker += exp_sat
                QSO_marker = ELG_marker
                if want_QSO:
                    M1_Q_temp = 10**(logM1_Q + As_Q * hdeltac[i] + Bs_Q * hfenv[i])
                    logM_cut_Q_temp = logM_cut_Q + Ac_Q * hdeltac[i] + Bc_Q * hfenv[i]
                    base_p_Q = N_sat_generic(
                        hmass[i], 10**logM_cut_Q_temp, kappa_Q, M1_Q_temp, alpha_Q, A_Q) * weights[i]
                    if enable_ranks:
                        decorator_Q = 1 + s_Q * ranks[i] + s_v_Q * ranksv[i] + s_p_Q * ranksp[i] + s_r_Q * ranksr[i]
                        exp_sat = base_p_Q * decorator_Q
                    else:
                        exp_sat = base_p_Q
                    QSO_marker += exp_sat

                if randoms[i] <= LRG_marker:
                    delsigma_L[tid, :, 0] += delsigma[i]
                    N_L[tid, 0] += 1
                elif randoms[i] <= ELG_marker:
                    delsigma_E[tid, :, 0] += delsigma[i]
                    N_E[tid, 0] += 1
                elif randoms[i] <= QSO_marker:
                    delsigma_Q[tid, :, 0] += delsigma[i]
                    N_Q[tid, 0] += 1

        delsigma_L = np.sum(delsigma_L, axis = 0)
        delsigma_E = np.sum(delsigma_E, axis = 0)
        delsigma_Q = np.sum(delsigma_Q, axis = 0)
        N_L = np.sum(N_L)
        N_E = np.sum(N_E)
        N_Q = np.sum(N_Q)

        return delsigma_L[:, 0], delsigma_E[:, 0], delsigma_Q[:, 0], N_L, N_E, N_Q

def main(path2config, mock_key = 'LRG', combo_key = 'LRG_LRG', load_bestfit = True):
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

    if load_bestfit:
        # load chains
        prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
                                    dynesty_config_params['chainsPrefix'])

        # datafile = np.load(prefix_chain+'_results.npz', allow_pickle = True)
        # res1 = datafile['res'].item()

        # # print the max logl fit
        # logls = res1['logl']
        # indmax = np.argmax(logls)
        # hod_params = res1['samples'][indmax]
        # print("max logl fit ", hod_params, logls[indmax])
        # hod_params = [12.63495771, 13.69554923, -2.37167367,  0.97122117,  1., 0.18353945,
        #                 1.00628539, -0.31160309,  0.41906906] # multipole B fit
        # hod_params = [12.70525188, 13.80426605, -2.79409949,  0.99480044,  1.,          0.20628246,
        #             1.00506506, -0.19501842,  0.31357795] # multipole B fit new
        # hod_params = [12.85769473, 14.13806258, -2.93011803,  0.93158762,  0.98706175,  0.21451697,
        #             0.95139146, -0.39688752,  0.88150003] # multipole A fit
        # hod_params = [12.89261965, 13.9752593,  -2.38207814,  1.05614104,  0.93290415,  0.12666648,
        # 0.92018434, -0.09396447,  0.09896801]
        hod_params = [13.00463651, 14.21416778, -5.21529117,  1.27676045, -0.25028219] # wp fit lowz

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
    delta_sig = newSigma.run_hod(Nthread = 64, verbose = True)[mock_key]
    for i in range(10):
        start = time.time()
        print(newSigma.run_hod(Nthread = 64, verbose = True), time.time() - start)

    np.savez("./data_lensing/data_"+dynesty_config_params['chainsPrefix'], rs = newData.rs[combo_key], delta_sig = delta_sig)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    DEFAULTS = {}
    DEFAULTS['path2config'] = 'config/boss_lrg_xi.yaml'

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    main(**args)

