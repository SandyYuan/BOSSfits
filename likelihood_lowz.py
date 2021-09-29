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
            clustering[key] = np.loadtxt(data_params['tracer_combos'][key]['path2power'])[:, 2]
        self.clustering = clustering

        # load the covariance matrix for all tracer combinations
        icov = {}
        diag = {}
        for key in data_params['tracer_combos'].keys():
            jacks = np.loadtxt(data_params['tracer_combos'][key]['path2jack'])[:, 2:]
            cov = np.cov(jacks)*(np.shape(jacks)[1]-1)
            diag[key] = np.sqrt(np.diag(cov))
            # normalize
            covnorm = np.zeros(np.shape(cov))
            for i in range(np.shape(cov)[0]):
                for j in range(np.shape(cov)[1]):
                    covnorm[i, j] = cov[i, j] / diag[key][i] / diag[key][j]
            icov[key] = np.linalg.inv(covnorm)
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
