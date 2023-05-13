from run_nerf_helpers import NeRF, Appendix, FMap

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

def generate_map(path, fine = False, appendix_0 = True):
    appendix = None
    with open(path, "rb") as f:
        appendix = pickle.load(f)[1] if fine else pickle.load(f)[0]
    appendix = appendix.appendix_0 if appendix_0 else appendix.appendix_1
    appendix.to('cuda')
    input_x1 = torch.arange(-1, 1, 0.01)*0.99
    input_x2 = torch.arange(-1, 1, 0.01)*0.99
    input_x1, input_x2 = torch.meshgrid(input_x1, input_x2)
    numel = input_x1.shape[0] * input_x1.shape[0]
    input_tensor = torch.hstack((input_x1.reshape((numel, 1)), input_x2.reshape((numel,1)))).to('cuda')
    output_tensor : torch.tensor = appendix(8*torch.arctanh(input_tensor))
    output_tensor = output_tensor.reshape((input_x1.shape[0], input_x1.shape[0], 3)).cpu().detach().numpy()
    
    return output_tensor

if __name__ =="__main__":
    generate_map("pickle_files/test/010000.pkl")
    