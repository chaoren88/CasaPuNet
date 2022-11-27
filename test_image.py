import os
import numpy as np
import torch
import argparse
import cv2
import glob
from CasaPuNet import Network


parser = argparse.ArgumentParser()
parser.add_argument('--ck_dir', type=str, default='./checkpoint/', help="Checkpoint directory")
parser.add_argument('--ck_name', type=str, default='checkpoint.pth.tar', help='Checkpoint name')
parser.add_argument('--input_dir', type=str, default='./image/', help='Test image directory')
parser.add_argument('--output_dir', type=str, default='./output/image/', help='Test result directory')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

net = Network()
net = torch.nn.DataParallel(net).cuda()

print('Loading the model ...')
checkpoint = torch.load(os.path.join(args.ck_dir, args.ck_name))
net.load_state_dict(checkpoint['state_dict'])

print('Starting the test ...')
net.eval()
input_fns = glob.glob(args.input_dir + '*.png')
input_fns.sort()
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
for i, input_fn in enumerate(input_fns):
    img = cv2.imread(input_fn)
    img = img[:, :, ::-1] / 255.0

    input_image = np.transpose(img, axes=[2, 0, 1]).astype('float32')
    input_var =  torch.from_numpy(input_image).unsqueeze(0).cuda()
    with torch.no_grad():
        _, output = net(input_var)

    output_image = np.transpose(output[0,...].cpu().numpy(), axes=[1, 2, 0]).astype('float32')
    output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[: ,: ,::-1]

    cv2.imwrite(os.path.join(args.output_dir, os.path.basename(input_fn)), output_image)

print('Finished!')
