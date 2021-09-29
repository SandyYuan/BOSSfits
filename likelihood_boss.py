import os

import numpy as np

class xirppi_Data(object):
    """
    Dummy object for calculating a likelihood
    """
    def __init__(self, data_params, HOD_params):
        """
        Constructor of the power spectrum data
        """
        num_dens_mean = {}
        num_dens_std = {}
        for key in HOD_params['tracer_flags'].keys():
            if HOD_params['tracer_flags'][key]:
                num_dens_mean[key] = data_params['tracer_density_mean'][key]
                num_dens_std[key] = data_params['tracer_density_std'][key]
        self.num_dens_mean = num_dens_mean
        self.num_dens_std = num_dens_std

        # load the power spectrum for all tracer combinations
        clustering = {}
        for key in data_params['tracer_combos'].keys():
            clustering[key] = np.loadtxt(data_params['tracer_combos'][key]['path2power']).reshape(1, -1)[0]
        self.clustering = clustering

        # load the covariance matrix for all tracer combinations
        icov = {}
        diag = {}
        for key in data_params['tracer_combos'].keys():
            cov = np.load(data_params['tracer_combos'][key]['path2cov'])['xicovnorm']
            icov[key] = np.linalg.inv(cov)
            diag[key] = np.sqrt(np.load(data_params['tracer_combos'][key]['path2cov'])['diag'])
        self.icov = icov
        self.diag = diag

    def compute_likelihood(self, theory_clustering, theory_density):
        """
        Computes the likelihood using information from the context
        """
        # Calculate a likelihood up to normalization
        lnprob = 0.
        for key in self.clustering.keys():
            delta = (self.clustering[key] - theory_clustering[key].flatten()) / self.diag[key]
            lnprob += np.einsum('i,ij,j', delta[6:], self.icov[key][6:, 6:], delta[6:])
            # print(self.clustering[key], theory_clustering[key].flatten())
            # print(delta)
        lnprob *= -0.5
        print("clustering lnprob", lnprob)

        # likelihood due to number density
        for etracer in self.num_dens_mean.keys():
            lnprob += -0.5*((self.num_dens_mean[etracer] - theory_density[etracer])/self.num_dens_std[etracer])**2

        # Return the likelihood
        print("theory density and target density", theory_density['LRG'], self.num_dens_mean['LRG'])
        print(" <><> Likelihood evaluated, lnprob = ",lnprob)
        return lnprob



class multipole_Data(object):
    """
    Dummy object for calculating a likelihood
    """
    def __init__(self, data_params, HOD_params):
        """
        Constructor of the power spectrum data
        """
        num_dens_mean = {}
        num_dens_std = {}
        for key in HOD_params['tracer_flags'].keys():
            if HOD_params['tracer_flags'][key]:
                num_dens_mean[key] = data_params['tracer_density_mean'][key]
                num_dens_std[key] = data_params['tracer_density_std'][key]
        self.num_dens_mean = num_dens_mean
        self.num_dens_std = num_dens_std

        # load the power spectrum for all tracer combinations
        multipoles = {}
        rs = {}
        for key in data_params['tracer_combos'].keys():
            wp = np.loadtxt(data_params['tracer_combos'][key]['path2wp'])[:, 1]
            xi0 = np.loadtxt(data_params['tracer_combos'][key]['path2xi0'])[:, 1]
            xi2 = np.loadtxt(data_params['tracer_combos'][key]['path2xi2'])[:, 1]
            xi4 = np.loadtxt(data_params['tracer_combos'][key]['path2xi4'])[:, 1]
            multipoles[key] = np.concatenate((wp, xi0, xi2, xi4))
            rs[key] = np.loadtxt(data_params['tracer_combos'][key]['path2xi0'])[:, 0]
        self.multipoles = multipoles
        self.rs = rs

        # load the covariance matrix for all tracer combinations
        icov = {}
        diag = {}
        for key in data_params['tracer_combos'].keys():
            covdata = np.loadtxt(data_params['tracer_combos'][key]['path2cov'])
            corr = np.zeros(np.shape(covdata))
            for i in range(np.shape(covdata)[0]):
                for j in range(np.shape(covdata)[1]):
                    corr[i, j] = covdata[i, j] / np.sqrt(covdata[i, i] * covdata[j, j])
            fullcorr_inv = np.linalg.inv(corr)
            icov[key] = fullcorr_inv
            diag[key] = np.sqrt(np.diag(covdata))
        self.icov = icov
        self.diag = diag

    def compute_likelihood(self, theory_clustering, theory_density):
        """
        Computes the likelihood using information from the context
        """
        lnprob = 0.
        for key in self.multipoles.keys():
            # print(self.multipoles[key], theory_clustering[key], self.rs)
            delta = (self.multipoles[key] - theory_clustering[key]) / self.diag[key]
            # print("rs", self.rs[key])
            # print("wp", self.rs[key]*self.multipoles[key][:8], self.rs[key]*theory_clustering[key][:8])
            # print("xi0", self.rs[key]**2*self.multipoles[key][8:16], self.rs[key]**2*theory_clustering[key][8:16])
            # print("xi2", self.rs[key]**2*self.multipoles[key][16:24], self.rs[key]**2*theory_clustering[key][16:24])
            # print("xi4", self.rs[key]**2*self.multipoles[key][24:32], self.rs[key]**2*theory_clustering[key][24:32])
            # print("dot prod", delta, self.icov[key], delta * np.dot(self.icov[key], delta))
            lnprob += np.einsum('i,ij,j', delta, self.icov[key], delta)
            # print(key, self.clustering[key], theory_clustering[key])
        lnprob *= -0.5
        print("clustering lnprob", lnprob)

        # likelihood due to number density
        for etracer in self.num_dens_mean.keys():
            lnprob += -0.5*((self.num_dens_mean[etracer] - theory_density[etracer])/self.num_dens_std[etracer])**2
            print(etracer, -0.5*((self.num_dens_mean[etracer] - theory_density[etracer])/self.num_dens_std[etracer])**2)

        # Return the likelihood
        print("theory density and target density", theory_density['LRG'], self.num_dens_mean['LRG'])
        print(" <><> Likelihood evaluated, lnprob = ",lnprob)
        return lnprob

class wp_Data(object):
    """
    Dummy object for calculating a likelihood
    """
    def __init__(self, data_params, HOD_params):
        """
        Constructor of the power spectrum data
        """
        num_dens_mean = {}
        num_dens_std = {}
        for key in HOD_params['tracer_flags'].keys():
            if HOD_params['tracer_flags'][key]:
                num_dens_mean[key] = data_params['tracer_density_mean'][key]
                num_dens_std[key] = data_params['tracer_density_std'][key]
        self.num_dens_mean = num_dens_mean
        self.num_dens_std = num_dens_std

        # load the power spectrum for all tracer combinations
        clustering = {}
        rs = {}
        for key in data_params['tracer_combos'].keys():
            clustering[key] = np.loadtxt(data_params['tracer_combos'][key]['path2power'])[:, 1] # wp
            rs[key] = np.loadtxt(data_params['tracer_combos'][key]['path2power'])[:, 0]
        self.clustering = clustering
        self.rs = rs

        # load the covariance matrix for all tracer combinations
        icov = {}
        for key in data_params['tracer_combos'].keys():
            covdata = np.loadtxt(data_params['tracer_combos'][key]['path2cov'])
            rs = np.loadtxt(data_params['tracer_combos'][key]['path2power'])[:, 0]
            hong_rwp_covmat = np.zeros(np.shape(covdata))
            for i in range(np.shape(covdata)[0]):
                for j in range(np.shape(covdata)[1]):
                    hong_rwp_covmat[i, j] = covdata[i, j]*rs[i]*rs[j]
            hong_rwp_covmat_inv = np.linalg.inv(hong_rwp_covmat)
            icov[key] = hong_rwp_covmat_inv
        self.icov = icov


    def compute_likelihood(self, theory_clustering, theory_density):
        """
        Computes the likelihood using information from the context
        """
        # Calculate a likelihood up to normalization
        lnprob = 0.
        for key in self.clustering.keys():
            delta = (self.rs[key] * (self.clustering[key] - theory_clustering[key]))[3:]
            lnprob += np.einsum('i,ij,j', delta, self.icov[key][3:, 3:], delta)
            # print(key, self.clustering[key], theory_clustering[key])
        lnprob *= -0.5
        print("clustering lnprob", lnprob)

        # likelihood due to number density
        for etracer in self.num_dens_mean.keys():
            lnprob += -0.5*((self.num_dens_mean[etracer] - theory_density[etracer])/self.num_dens_std[etracer])**2
            print(etracer, -0.5*((self.num_dens_mean[etracer] - theory_density[etracer])/self.num_dens_std[etracer])**2)

        # Return the likelihood
        print("theory density and target density", theory_density['LRG'], self.num_dens_mean['LRG'])
        print(" <><> Likelihood evaluated, lnprob = ",lnprob)
        return lnprob

class sigma_Data(object):
    def __init__(self, data_params, HOD_params):
        num_dens_mean = {}
        num_dens_std = {}
        for key in HOD_params['tracer_flags'].keys():
            if HOD_params['tracer_flags'][key]:
                num_dens_mean[key] = data_params['tracer_density_mean'][key]
                num_dens_std[key] = data_params['tracer_density_std'][key]
        self.num_dens_mean = num_dens_mean
        self.num_dens_std = num_dens_std

        deltasigma = {}
        covs = {}
        icovs = {}
        rs = {}
        rbins = {}
        for key in data_params['tracer_combos'].keys():
            deltasigma[key] = np.loadtxt(data_params['tracer_combos'][key]['path2sigma'])[:,1] # h Msun pc-2]
            rs[key] = np.loadtxt(data_params['tracer_combos'][key]['path2sigma'])[:,0] # h-1 mpc
            delta_r_log = (np.log10(rs[key][-1]) - np.log10(rs[key][0])) / (len(rs[key]) - 1)
            rbins[key] = 10**(np.linspace(np.log10(rs[key][0]) - delta_r_log/2, np.log10(rs[key][-1]) + delta_r_log/2, len(rs[key]) + 1))

            covmat_data = np.loadtxt(data_params['tracer_combos'][key]['path2sigmacov'])
            covmat = np.zeros((len(rs[key]), len(rs[key])))
            k = 0
            for i in range(len(rs[key])):
                for j in range(len(rs[key])):
                    covmat[i, j] = covmat_data[k]
                    k += 1
            covs[key] = covmat
            icovs[key] = np.linalg.inv(covmat)

        self.deltasigma = deltasigma
        self.covs = covs
        self.icovs = icovs
        self.rs = rs
        self.rbins = rbins


    def compute_likelihood(self, theory_sigma, theory_density):
        """
        Computes the likelihood using information from the context
        """
        # Calculate a likelihood up to normalization
        delta = (self.deltasigma['LRG_LRG'] - theory_sigma['LRG']/1e12)
        lnprob = -0.5*np.einsum('i,ij,j', delta, self.icovs['LRG_LRG'], delta)
        # print(key, self.clustering[key], theory_clustering[key])
        print("deltasigma lnprob", lnprob)

        # likelihood due to number density
        for etracer in self.num_dens_mean.keys():
            lnprob += -0.5*((self.num_dens_mean[etracer] - theory_density[etracer])/self.num_dens_std[etracer])**2
            print(etracer, -0.5*((self.num_dens_mean[etracer] - theory_density[etracer])/self.num_dens_std[etracer])**2)

        # Return the likelihood
        print("theory density and target density", theory_density['LRG'], self.num_dens_mean['LRG'])
        print(" <><> Likelihood evaluated, lnprob = ",lnprob)
        return lnprob



