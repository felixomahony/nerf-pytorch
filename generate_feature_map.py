from run_nerf_helpers import NeRF, Appendix, FMap

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

def generate_f_map(path, fmap_0 = True):
    fmap = None
    with open(path, "rb") as f:
        fmap = pickle.load(f)[0]
    fmap : FMap = fmap.fmap_0 if fmap_0 else fmap.fmap_1
    fmap.to('cpu')
    with open("inputs_ex.pkl", "rb") as f:
        iputs = pickle.load(f)
    iputs.to('cpu')
    output_tensor = fmap(iputs)

    
    return output_tensor