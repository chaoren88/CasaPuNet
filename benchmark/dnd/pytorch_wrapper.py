 # Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

 # This file is part of the implementation as described in the CVPR 2017 paper:
 # Tobias Plötz and Stefan Roth, Benchmarking Denoising Algorithms with Real Photographs.
 # Please see the file LICENSE.txt for the license governing this code.

import numpy as np
import torch
from torch.autograd import Variable

def pytorch_denoiser(denoiser, use_cuda):
    def wrap_denoiser(Inoisy, nlf=None):
        if len(Inoisy.shape) > 2:
            Inoisy = np.transpose(Inoisy, axes=[2, 0, 1])
        noisy = torch.from_numpy(Inoisy)
        if len(noisy.shape) > 2:
            noisy = noisy.unsqueeze(0)
        else:
            noisy = noisy.view(1,1, noisy.shape[0], noisy.shape[1])
        if use_cuda:
            noisy = noisy.cuda()

        noisy = Variable(noisy)
        with torch.no_grad():
            _, denoised = denoiser(noisy)

        denoised = denoised[0,...].cpu().numpy()
        denoised = np.transpose(denoised, [1,2,0])
        return denoised

    return wrap_denoiser


