from run_nerf_helpers import NeRF, Appendix, FMap

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

def get_multivariate_gaussian(path):
    network = None
    with open(path, "rb") as f:
        network = pickle.load(f)[0]
    fmap_0 = network.fmap_0
    fmap_0.to('cpu')
    fmap_1 = network.fmap_1
    fmap_1.to('cpu')

    with open("inputs_ex.pkl", "rb") as f:
        iputs = pickle.load(f)
    iputs.to('cpu')

    out_0 = fmap_0(iputs)
    out_1 = fmap_1(iputs)

    mask_0 = out_0[0][:,0].cpu().detach().numpy() > 0.5
    mask_1 = out_1[0][:,0].cpu().detach().numpy() > 0.5
    #Note here that the masking is only to allow 
    output_tensor_0 = np.stack((out_0[1].cpu().detach().numpy()[mask_0,0], out_0[1].cpu().detach().numpy()[mask_0,1]),axis=1).T
    output_tensor_0 = output_tensor_0 / output_tensor_0.shape[0]
    output_tensor_1 = np.stack((out_1[1].cpu().detach().numpy()[mask_1,0], out_1[1].cpu().detach().numpy()[mask_1,1]),axis=1).T
    output_tensor_1 = output_tensor_1 / output_tensor_1.shape[0]

    # output_tensor_0 = out_0[1].cpu().detach().numpy()[np.repeat(out_0[0].cpu().detach().numpy(),2,axis=1) > 0.5].T
    # output_tensor_1 = out_1[1].cpu().detach().numpy()[np.repeat(out_1[0].cpu().detach().numpy(),2,axis=1) > 0.5].T

    cov_0 = np.cov(output_tensor_0)
    cov_1 = np.cov(output_tensor_1)

    mu_0 = np.mean(output_tensor_0, axis=1)
    mu_1 = np.mean(output_tensor_1, axis=1)
    
    return (cov_0, cov_1, mu_0, mu_1)

def get_kl_divergence(path):
    cov_0, cov_1, mu_0, mu_1 = get_multivariate_gaussian(path)

    #This eqn is taken from https://stanford.edu/~jduchi/projects/general_notes.pdf

    term_0 = np.log(np.linalg.det(cov_1)/np.linalg.det(cov_0))
    term_1 = -mu_0.shape[0]
    term_2 = np.trace(np.linalg.inv(cov_1) @ cov_0)
    term_3 = (mu_1-mu_0).T @ np.linalg.inv(cov_1) @ (mu_1 - mu_0)
    print(term_0, term_1, term_2, term_3)
    kl_div = 0.5 * (term_0+term_1+term_2+term_3)
    return kl_div

# if __name__ =="__main__":
#     get_kl_divergence("pickle_files/test/010000.pkl")
    