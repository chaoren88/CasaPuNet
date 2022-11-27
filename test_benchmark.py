from benchmark import dnd, sidd
import torch
from CasaPuNet import Network
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--ck_dir', type=str, default='./checkpoint/', help="Checkpoint directory")
parser.add_argument('--ck_name', type=str, default='checkpoint.pth.tar', help='Checkpoint name')
parser.add_argument('--type', type=str, default='dnd', help='To choose benchmark dataset, SIDD or DND')
parser.add_argument('--dnd_noisy_dir', type=str, default='./dataset/benchmark/dnd_2017/', help='DND benchmark dataset directory')
parser.add_argument('--output_dir', type=str, default='./output/', help='DND test result directory')
parser.add_argument('--sidd_noisy_file', type=str, default='./dataset/benchmark/sidd/BenchmarkNoisyBlocksSrgb.mat',
                        help="SIDD benchmark mat file")
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
output_dir = os.path.join(args.output_dir, args.type)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
if args.type == 'dnd':
    denoiser = dnd.pytorch_denoiser(net, use_cuda=True)
    dnd.denoise_srgb(denoiser, args.dnd_noisy_dir, output_dir)
    dnd.bundle_submissions_srgb(output_dir)
elif args.type == 'sidd':
    sidd.test(net, args.sidd_noisy_file, output_dir)

print('Finished!')
