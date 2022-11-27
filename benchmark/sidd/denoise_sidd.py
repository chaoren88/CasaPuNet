import os
import numpy as np
from skimage import img_as_float,img_as_ubyte
from tqdm import tqdm
from scipy.io import loadmat, savemat
import torch


def denoise(model, noisy_image):
    with torch.autograd.set_grad_enabled(False):
        noisy_image = noisy_image.cuda()

        _, phi_Z =  model(noisy_image)
        im_denoise = phi_Z.cpu().numpy()

    im_denoise = np.transpose(im_denoise.squeeze(), (1, 2, 0))
    im_denoise = img_as_ubyte(im_denoise.clip(0, 1))

    return im_denoise


def test(model, noisy_mat_file, output_dir):
    noisy_data_mat_file = noisy_mat_file
    noisy_data_mat_name = os.path.basename(noisy_data_mat_file).replace('.mat', '')
    noisy_data_mat = loadmat(noisy_data_mat_file)[noisy_data_mat_name]

    npose = (noisy_data_mat.shape[0])
    nsmile = noisy_data_mat.shape[1]
    poseSmile_cell = np.empty((npose, nsmile), dtype=object)

    for image_index in tqdm(range(noisy_data_mat.shape[0])):
        for block_index in range(noisy_data_mat.shape[1]):
            noisy_image = noisy_data_mat[image_index, block_index, :, :, :]
            noisy_image = np.float32(noisy_image / 255.)
            noisy_image = torch.from_numpy(noisy_image.transpose((2, 0, 1))[np.newaxis,])
            poseSmile_cell[image_index,block_index] = denoise(model, noisy_image)

    submit_data = {
            'DenoisedBlocksSrgb': poseSmile_cell
        }

    savemat(
            os.path.join(output_dir, "SubmitSrgb.mat"),
            submit_data
        )
