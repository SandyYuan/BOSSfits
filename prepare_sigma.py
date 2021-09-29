"""
load subsampled halo and particle catalogs and compute their sigmas
"""

import os
import glob
import time
import timeit
from pathlib import Path
import yaml

import numpy as np
import h5py
import asdf
import argparse
from itertools import repeat
import multiprocessing
from multiprocessing import Pool
from astropy.io import ascii

from abacusnbody.hod.abacus_hod import AbacusHOD
# from likelihood_boss import sigma_Data

from halotools.mock_observables import mean_delta_sigma

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

def prepare_slab_sigma(i, savedir,  MT, want_ranks, subsample, newseed, lbox, Mpart, rbins):
    outfilename_halos = savedir+'/halos_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushod_oldfenv'
    outfilename_particles = savedir+'/particles_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushod_oldfenv'
    print("processing slab ", i)
    if MT:
        outfilename_halos += '_MT'
        outfilename_particles += '_MT'
    if want_ranks:
        outfilename_particles += '_withranks'
    outfilename_particles += '_new'
    outfilename_halos += '_new'

    if not (os.path.exists(outfilename_halos+'.h5') \
    and os.path.exists(outfilename_particles+'.h5')):
        raise ValueError("halo or particle file doesnt exist")

    # load the halo and particle subsamples 
    newfile = h5py.File(outfilename_halos+'.h5', 'r')
    newpart = h5py.File(outfilename_particles+'.h5', 'r')
    subhalos = newfile['halos']
    subparts = newpart['particles']
    print(len(subhalos), len(subparts))

    print("starting halos")
    start = time.time()
    delta_sig_halos = mean_delta_sigma(subhalos['x_L2com'] % lbox, subsample, Mpart, 
        rbins, lbox, per_object = True)
    print("finished halos. took time ", time.time() - start)

    newfile = h5py.File(outfilename_halos+'_sigma.h5', 'w')
    dataset = newfile.create_dataset('sigma', data = delta_sig_halos)
    newfile.close()

    print("starting particles")
    start = time.time()
    delta_sig_parts = mean_delta_sigma(subparts['pos'] % lbox, subsample, Mpart, 
        rbins, lbox, per_object = True)
    print("finished finished. took time ", time.time() - start)

    newfile = h5py.File(outfilename_particles+'_sigma.h5', 'w')
    dataset = newfile.create_dataset('sigma', data = delta_sig_parts)
    newfile.close()
    print("finished slab ", i)


def main(path2config, params = None, mock_key = 'LRG'):
    print("compiling compaso halo catalogs into subsampled catalogs")

    config = yaml.load(open(path2config))
    # update params if needed
    if params is None:
        params = {}
    config.update(params)

    simname = config['sim_params']['sim_name'] # "AbacusSummit_base_c000_ph006"
    simdir = config['sim_params']['sim_dir']
    z_mock = config['sim_params']['z_mock']
    savedir = config['sim_params']['subsample_dir']+simname+"/z"+str(z_mock).ljust(5, '0') 
    cleaning = config['sim_params']['cleaned_halos']
    downsample = config['sigma_params']['downsample']

    # run a sample hod to get some sim parameters
    newBall = AbacusHOD(config['sim_params'], config['HOD_params'], config['clustering_params'])
    lbox = newBall.lbox
    Mpart_eff = newBall.params['Mpart'] / 0.03 / downsample # msun / h
    # newData = sigma_Data(config['data_params'], config['HOD_params'])
    # rbins = newData.rbins[mock_key+'_'+mock_key]
    rbins =  np.array(config['sigma_params']['rbins'])


    halo_info_fns = \
    list((Path(simdir) / Path(simname) / 'halos' / ('z%4.3f'%z_mock) / 'halo_info').glob('*.asdf'))
    numslabs = len(halo_info_fns)

    # particle subsample directory
    subsample_name = Path(config['sim_params']['subsample_dir']) / simname / ('z%4.3f'%z_mock) \
    / ('parts_all_down_%4.3f.h5'%downsample)
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

    full_subsample = full_subsample % lbox

    # MT flag
    tracer_flags = config['HOD_params']['tracer_flags']
    MT = False
    if tracer_flags['ELG'] or tracer_flags['QSO']:
        MT = True
    want_ranks = config['HOD_params']['want_ranks']
    want_AB = config['HOD_params']['want_AB']
    newseed = 600

    # compute sigma 
    # prepare_slab_sigma(0, savedir, MT, want_ranks, full_subsample, newseed, lbox, Mpart_eff, rbins)

    # for i in range(1, numslabs):
    #     prepare_slab_sigma(i, savedir, MT, want_ranks, full_subsample, newseed, lbox, Mpart_eff, rbins)

    p = multiprocessing.Pool(config['prepare_sim']['Nparallel_load'])
    p.starmap(prepare_slab_sigma, zip(range(numslabs), repeat(savedir), 
        repeat(MT), repeat(want_ranks), repeat(full_subsample), repeat(newseed),
        repeat(lbox), repeat(Mpart_eff), repeat(rbins)))
    p.close()
    p.join()



class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    main(**args)
