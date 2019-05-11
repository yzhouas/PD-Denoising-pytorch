import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import *
from utils import *
from denoiser import *
from PIL import Image
import scipy.io as sio

parser = argparse.ArgumentParser(description="PD-denoising")
#model parameter
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--delog", type=str, default="logsdc", help='path of log and model files')
parser.add_argument("--mode", type=str, default="M", help='DnCNN-B (B) or MC-AWGN-RVIN (MC)')
#tested noise type
parser.add_argument("--color", type=int, default=0, help='[0]gray [1]color')
parser.add_argument("--real_n", type=int, default=0, help='real noise or synthesis noise [0]synthetic noises [1]real noisy image wo gnd [2]real noisy image with gnd')
parser.add_argument("--spat_n", type=int, default=0, help='whether to add spatial-variant signal-dependent noise, [0]no spatial [1]Gaussian-possion noise')
#pixel-shuffling parameter
parser.add_argument("--ps", type=int, default=0, help='pixel shuffle [0]no pixel-shuffle [1]adaptive pixel-ps [2]pre-set stride')
parser.add_argument("--wbin", type=int, default=512, help='patch size while testing on full images')
parser.add_argument("--ps_scale", type=int, default=2, help='if ps==2, use this pixel shuffle stride')
#down-scaling parameter
parser.add_argument("--scale", type=float, default=1, help='resize the original images')
parser.add_argument("--rescale", type=int, default=1, help='resize it back to the origianl size after downsampling')
#testing data path and processing
parser.add_argument("--test_data", type=str, default='Set12', help='testing data path')
parser.add_argument("--test_data_gnd", type=str, default='Set12', help='testing data ground truth path if it exists')
parser.add_argument("--cond", type=int, default=1, help='Testing mode using noise map of: [0]Groundtruth [1]Estimated [2]External Input')
parser.add_argument("--test_noise_level", nargs = "+",  type=int, help='input noise level while generating noisy images')
parser.add_argument("--ext_test_noise_level", nargs = "+", type=int, help='external noise level input used if cond==2')
#refining on the estimated noise map
parser.add_argument("--refine", type=int, default=0, help='[0]no refinement of estimation [1]refinement of the estimation')
parser.add_argument("--refine_opt", type=int, default=0, help='[0]get the most frequent [1]the maximum [2]Gaussian smooth [3]average value of 0 and 1 opt')
parser.add_argument("--zeroout", type=int, default=0, help='[0]no zeroing out [1]zeroing out some maps')
parser.add_argument("--keep_ind", nargs = "+", type=int, help='[0 1 2]Gaussian [3 4 5]Impulse')
#output options
parser.add_argument("--output_map", type=int, default=0, help='whether to output maps')
parser.add_argument("--k", type=float, default=1, help='merging factor between details and background')
parser.add_argument("--out_dir", type=str, default="results_bc", help='path of output files')

opt = parser.parse_args()
#the limitation range of each type of noise level: [0]Gaussian [1]Impulse
limit_set = [[0,75], [0, 80]]

def img_normalize(data):
    return data/255.

def main():
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    # Build model
    print('Loading model ...\n')
    c = 1 if opt.color == 0 else 3
    net = DnCNN_c(channels=c, num_of_layers=opt.num_of_layers, num_of_est = 2 * c)
    est_net = Estimation_direct(c, 2 * c)

    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.delog, 'net.pth')))
    model.eval()

    #Estimator Model
    model_est = nn.DataParallel(est_net, device_ids=device_ids).cuda()
    model_est.load_state_dict(torch.load(os.path.join(opt.delog, 'est_net.pth')))
    model_est.eval()

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.*'))
    files_source.sort()

    #process images with pre-defined noise level
    for f in files_source:
        print(f)
        file_name = f.split('/')[-1].split('.')[0]
        if opt.real_n == 2:  #have ground truth
            gnd_file_path = os.path.join('data',opt.test_data_gnd, file_name + '_mean.png')
            print(gnd_file_path)
            Img_gnd = cv2.imread(gnd_file_path)
            Img_gnd = Img_gnd[:,:,::-1]
            Img_gnd = cv2.resize(Img_gnd, (0,0), fx=opt.scale, fy=opt.scale)
            Img_gnd = img_normalize(np.float32(Img_gnd))
        # image
        Img = cv2.imread(f)  #input image with w*h*c
        w, h, _ = Img.shape
        Img = Img[:,:,::-1]  #change it to RGB
        Img = cv2.resize(Img, (0,0), fx=opt.scale, fy=opt.scale)
        if opt.color == 0:
            Img = Img[:,:,0]  #For gray images
            Img = np.expand_dims(Img, 2)
        pss=1
        if opt.ps == 1:
            pss = decide_scale_factor(Img/255., model_est, color=opt.color,  thre = 0.008, plot_flag = 1, stopping = 4,mark = opt.out_dir + '/' +  file_name)[0]
            print(pss)
            Img = pixelshuffle(Img, pss)
        elif opt.ps == 2:
            pss = opt.ps_scale
        
        merge_out= np.zeros([w,h,3])
        print('Splitting and Testing.....')
        wbin = opt.wbin
        i = 0
        while i < w:
            i_end = min(i+wbin, w)
            j = 0
            while j < h:
                j_end = min(j+wbin, h)
                patch = Img[i:i_end,j:j_end,:]
                patch_merge_out_numpy = denoiser(patch, c, pss, model, model_est, opt)
                merge_out[i:i_end, j:j_end, :] = patch_merge_out_numpy        
                j = j_end
            i = i_end
        cv2.imwrite(os.path.join(opt.out_dir, file_name + '_pss'+str(pss)+'_k'+str(opt.k)+'.png'), merge_out[:,:,::-1])
        print('done!')


if __name__ == "__main__":
    main()
