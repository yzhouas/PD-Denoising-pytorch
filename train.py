import os
import itertools
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import *
from dataset import prepare_data, Dataset
from utils import *
from PIL import Image
import concurrent.futures

#Basic Configuration
#noise type to consider: Gaussian, Uniform, Possion, Salt and Pepper
parser = argparse.ArgumentParser(description="mix-DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="MC", help='DnCNN-B (B) or AWGN-RVIN-MC(MC)')
parser.add_argument("--color", type=int, default=1, help='DnCNN-B (B) or our mixed-denoisor(M)')


opt = parser.parse_args()
#the limitation range of each type of noise level: [0]Gaussian [1]Impulse
limit_set = [[0,75], [0, 80]]
    
def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(opt.color, train=True)
    #dataset_val = Dataset(opt.color, train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=16, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    c = 1 if opt.color == 0 else 3
    # Build model
    if opt.mode == "MC":
        net = DnCNN_c(channels=c, num_of_layers=opt.num_of_layers, num_of_est = 2 * c)  #denoisor
        est_net = Estimation_direct(c, 2 * c)  #estimator
        est_net.apply(weights_init_kaiming)

    elif opt.mode == "B":
        net = DnCNN(channels=c, num_of_layers=opt.num_of_layers)

    #weights initialization
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    #criterion_class = As2DCriterion(nn.CrossEntropyLoss())
    # Move to GPU
    criterion.cuda()
    #criterion_class.cuda()
    model = nn.DataParallel(net).cuda()

    if opt.mode == "MC":
        est_model = nn.DataParallel(est_net).cuda()
        optimizer = optim.Adam(itertools.chain(est_model.parameters(), model.parameters()), lr=opt.lr)  #train all the parameter together    
    elif opt.mode == "B":
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0

    for epoch in range(opt.epochs):
        
        #start traning: adjusting learning rate
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            if opt.mode == "MC":
                est_model.train()
                est_model.zero_grad()
            model.zero_grad()
            optimizer.zero_grad()

            #data loading
            img_train = data.clone() #the original clean images: for the ground truth
            imgn_train = data.clone()  #the initialized noisy images: for the contaminative data
            
            sizeN = img_train.size()
            patch_n, patch_c, patch_w, patch_h = sizeN #the the patch size
            noise_map = np.zeros((patch_n, 2*c, patch_w, patch_h))  #initialize the noise map before concatenating
            
            for n in range(patch_n):  #go through all the images and create the noise to it   
                current_img = np.array(img_train[n, :, :, :].cpu().numpy())  #get the current image #c*w*h
                current_img = current_img.transpose(1,2,0)  #w*h*c
                #choose mixed or single
                s_or_m = np.random.randint(2) #choose single (0) or mixed(1)
                #set up noise
                noise_level_list = np.zeros((2*c, 1))
                #single noise type
                if s_or_m == 0:  #single noise type
                    noise_type = np.random.randint(2)  #choose just one type of noise to contaminate the image, AWGN or RVIN, return 0,1 
                    #multi-channel single type noises
                    for chn in range(c):
                        noise_level_list[chn + noise_type * c] = np.random.uniform(limit_set[noise_type][0], limit_set[noise_type][1])
                    noisy_img = generate_noisy(current_img, noise_type, noise_level_list[noise_type * c : noise_type * c + c]/255.)  #/255 to get the std between [0,1] or the ratio
                #mixed Multi-channel Noises (AWGN+RVIN)
                elif s_or_m == 1:
                    for noise_type in range(2):  #noise type
                        for chn in range(c):
                            noise_level_list[chn + noise_type * c] = np.random.uniform(limit_set[noise_type][0], limit_set[noise_type][1])
                    #noisy_img, p_i, p_m = generate_comp_noisy(current_img, noise_level_list[1]/255., 0, noise_level_list[0]/255.)
                    noisy_img = generate_comp_noisy(current_img, noise_level_list/255.)
                #noise_type_target = np.argmax(position_map, axis=1)  #get the position target positions
                imgn_train[n, :, :, :] = np2ts(noisy_img)
                #start normalize the noise level list to generate noise estimation ground truth map
                if opt.mode == "MC":
                    for noise_type in range(2):
                        for chn in range(c):
                            noise_level_list[noise_type * c + chn] = normalize(noise_level_list[noise_type * c + chn], 1, limit_set[noise_type][0], limit_set[noise_type][1])  #normalize the level value
                    noise_map[n, :, :, :] = np.reshape(np.tile(noise_level_list, patch_w * patch_h), (2*c, patch_w, patch_h))  #total number of channels
                    NM_tensor = torch.from_numpy(noise_map).type(torch.FloatTensor)
                    NM_tensor = Variable(NM_tensor.cuda())

             
            #Then clamp the image to a valid range [0,1]
            imgn_train = torch.clamp(imgn_train, 0., 1.)  #clamp the noisy image from 0-1
            residual = imgn_train - img_train
            
            #Change it to cuda variable
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            residual = Variable(residual.cuda())
            
        
            #put noisy image to the model: NL_tensor_map is a concatenated version of maps
            if opt.mode == "MC":
                outest = torch.clamp(est_model(imgn_train), 0., 1.)
                outres_train = model(imgn_train, outest)
            elif opt.mode == "B":
                outres_train = model(imgn_train)


            loss = criterion(outres_train, residual) / (img_train.size()[0]*2)
            if opt.mode == "MC":
                est_loss = criterion(outest, NM_tensor) / (img_train.size()[0] * 3 * img_train.size()[2] * img_train.size()[3] * 2)  #estimating the intensity
                den_full_loss = loss + est_loss
            elif opt.mode == 'B':
                den_full_loss = loss
            den_full_loss.backward()
            optimizer.step()

            #for a cycle-consistent estimation (update the ground truth input)
            if opt.mode == "MC":
                model.zero_grad()
                est_model.zero_grad()
                optimizer.zero_grad()

                gen_outres_train = model(imgn_train, NM_tensor)  #the generated noisy image using the ground label
                gen_loss = criterion(gen_outres_train, residual) / ((img_train.size()[0]*2))
                gen_full_loss = gen_loss
                gen_full_loss.backward()
                optimizer.step()

            

            # results to evaluate a batch results
            model.eval()
            if opt.mode == "MC":
                est_model.eval()
                outest = est_model(imgn_train)
                #outest = torch.cat([outnti, outnt], dim=1)
                outres_train = model(imgn_train, torch.clamp(outest, 0., 1.))
            elif opt.mode == "B":
                outres_train = model(imgn_train)

            out_train = torch.clamp(imgn_train-outres_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            if opt.mode == 'M' or opt.mode == "MC":
                print("[epoch %d][%d/%d] est_loss: %.4f loss: %.4f gen_loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), est_loss.item(), loss.item(), gen_loss.item(), psnr_train))
            elif opt.mode =='B':
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))

            if step % 10 == 0:
                # log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                if opt.mode == 'M' or opt.mode == "MC":
                    writer.add_scalar('estimation_loss', est_loss.item(), step)
                    writer.add_scalar('ground_denoising_loss', gen_loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            '''
            if step % 100 == 0:
                model.eval()
                if opt.mode == "MC":
                     est_model.eval()
                psnr_val_b = 0
                psnr_val_nb = 0
                crit = 0.
                #single image validation version, one can change it to the validation set
                for k in range(1):
                    img_val = cv2.imread('data/Set12/12.png')
                    img_w, img_h, _ = w, h, _ = img_val.shape
                    if opt.color == 0:
                       img_val = img_val[:,:,0]
                       img_val = np.expand_dims(img_val, 2)
                    img_val = np.float32(img_val) / 255.
                    noise_map_val = np.zeros((1, 2*c, img_w, img_h))
                    #TODO: Generate Noisy Image
                    imgn_val, noise_level_list_val = generate_training_noisy_image(img_val, 0, limit_set, c, 1)
                    img_val = np2ts(img_val)
                    imgn_val = np2ts(imgn_val)
                    #generate noise estimation ground truth map
                    if opt.mode == "MC":
                        noise_map_val = generate_ground_truth_noise_map(noise_map_val, 0, noise_level_list_val, limit_set, c, 1, img_w, img_h)
                        NM_tensor_val = torch.from_numpy(noise_map_val).type(torch.FloatTensor)
                        NM_tensor_val = Variable(NM_tensor_val.cuda())
                    img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
                    if opt.mode == "MC":
                        est_model.eval()
                        outest = torch.clamp(est_model(imgn_val), 0., 1.)
                        out_val_b = torch.clamp( imgn_val - model(imgn_val, outest), 0., 1.)
                        out_val_nb = torch.clamp( imgn_val - model(imgn_val, NM_tensor_val), 0., 1.)
                        psnr_val_nb += batch_PSNR(out_val_nb, img_val, 1.)
                    elif opt.mode == "B":
                        out_val_b = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
                        psnr_val_nb = 0

                    #crit += evl_criterion(out_val_b, img_val).item()
                    psnr_val_b += batch_PSNR(out_val_b, img_val, 1.)

                print("\n[val at epoch %d] PSNR_val_b: %.4f, PSNR_val_nb: %.4f" % (epoch+1, psnr_val_b, psnr_val_nb))

                writer.add_scalar('PSNR on validation data (blind)', psnr_val_b, epoch*len(loader_train) + i)
                writer.add_scalar('PSNR on validation data (non_blind)', psnr_val_nb, epoch*len(loader_train) + i)
            '''
            step += 1
            
        ## the end of each epoch
        # save model
        if not os.path.exists(opt.outf):
            os.makedirs(opt.outf)
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
        if opt.mode == "MC":
            torch.save(est_model.state_dict(), os.path.join(opt.outf, 'est_net.pth'))

if __name__ == "__main__":
    if opt.preprocess==1:
        prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2, color=opt.color)
    main()
