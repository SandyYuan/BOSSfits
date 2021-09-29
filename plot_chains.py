import os

import numpy as np
import matplotlib.pyplot as plt
import argparse
import getdist
from getdist import plots, MCSamples
import yaml

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/boss_lrg_wp.yaml'

def get_samples(outfile, par_names, w_rat, n_par, b_iter):
    marg_chains = np.loadtxt(outfile)
    # uncomment for when your chains have been complete
    #marg_chains = marg_chains[w_rat*n_par*b_iter:]
    marg_chains = marg_chains[3*marg_chains.shape[0]//4:]
    hsc = MCSamples(samples=marg_chains, names=par_names)
    return hsc

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n = 2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(y)
    print(f)
    taus = 2.0 * np.cumsum(f) - 1.0
    print(taus)
    window = auto_window(taus, c)
    print(window)
    return taus[window]

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def main(path2config):
    # read parameters
    config = yaml.load(open(path2config))
    fit_params = config['fit_params']
    ch_params = config['ch_config_params']

    # parameters
    n_iter = ch_params['sampleIterations']
    w_rat = ch_params['walkersRatio']
    b_iter = ch_params['burninIterations']
    par_names = fit_params.keys()
    lab_names = par_names
    n_par = len(par_names)

    # prior means
    prior_means = {}
    for ekey in fit_params.keys():
        prior_means[ekey] = fit_params[ekey][1]

    # what are we plotting
    HOD_pars = par_names
    filename = "plots/triangle_plot_"+ch_params['chainsPrefix']+".png"
    dir_chains = ch_params['path2output']

    # walkers ratio, number of params and burn in iterations
    marg_outfile = os.path.join(dir_chains, (ch_params['chainsPrefix']+".txt"))

    # read the samples after removing burnin
    marg_hsc = get_samples(marg_outfile, par_names, w_rat, n_par, b_iter)
    # marg_hsc.samples[:, -2] = abs(marg_hsc.samples[:, -2])

    mychain = marg_hsc.samples[:, 3]
    print(autocorr_gw2010(mychain))
    fig = plt.figure(figsize = (15, 4))
    plt.plot(mychain[10000:10000+1000], alpha = 0.4)
    plt.tight_layout()
    fig.savefig("plots/test_samples.png")

    # Triangle plot
    g = plots.getSubplotPlotter()
    g.settings.legend_fontsize = 20
    g.settings.scaling_factor = 0.1
    g.triangle_plot([marg_hsc], params=HOD_pars, filled = True, title_limit = 1, markers = prior_means)
    plt.savefig(filename)
    plt.close()

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    main(**args)

