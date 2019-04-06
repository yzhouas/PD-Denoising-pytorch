import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import *

def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

#color=0 for gray, color=1 for color images
def prepare_data(data_path, patch_size, stride, aug_times=1, color=0):
    # train
    print('process training data')
    #scales = [1, 0.9, 0.8, 0.7]
    scales = [1]
    if color == 0:
        files = glob.glob(os.path.join(data_path, 'train', '*.png'))
        h5f = h5py.File('train.h5', 'w')
    elif color == 1:
        files = glob.glob(os.path.join(data_path, 'train_c', '*.jpg'))
        h5f = h5py.File('train_c.h5', 'w')
    files.sort()
    train_num = 0
    for i in range(len(files)):
        print(i)
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            if color == 0:
                Img = np.expand_dims(Img[:,:,0].copy(), 0)
            elif color == 1:
                Img = np.transpose(Img, (2,0,1)) #move the channel to the first dimension
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    #files.clear()
    files = []
    files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)


#Prepare the data for real image and noise
def prepare_real_data(real_data_path, noise_data_path, patch_size, stride, aug_times=1, color=0):
    # train
    print('process training data')
    #scales = [1, 0.9, 0.8, 0.7]
    scales = [1]
    if color == 0:
        real_files = glob.glob(os.path.join(real_data_path, 'train', '*.png'))
        noise_files = glob.glob(os.path.join(noise_data_path, 'train', '*png'))
        h5f = h5py.File('train.h5', 'w')
    elif color == 1:
        real_files = glob.glob(os.path.join(real_data_path, 'train_c', '*.jpg'))
        noise_files = glob.glob(os.path.join(noise_data_path, 'train_c', '*jpg'))
        h5f = h5py.File('train_c.h5', 'w')
    files.sort()
    train_num = 0
    for i in range(len(real_files)):
        real_img = cv2.imread(real_files[i])
        noise_img = cv2.imread(noise_files[i])
        h, w, c = real_img.shape
        for k in range(len(scales)):
            Img = cv2.resize(real_img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            NImg = cv2.resize(noise_img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            if color == 0:
                Img = np.expand_dims(Img[:,:,0].copy(), 0)
                NImg = np.expend_dims(NImg[:,:,0].copy(), 0)
            elif color == 1:
                Img = np.transpose(Img, (2,0,1)) #move the channel to the first dimension
                NImg = np.transpose(NImg, (2,0,1))
            Img = np.float32(normalize(Img))
            NImg = np.float32(normalize(NImg))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            Npatches = Im2Patch(NImg, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (real_files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                ndata = Npatches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                h5f.create_dataset(str(train_num)+"_noise", data=ndata)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    ndata_aug = data_augmentation(ndata, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    h5f.create_dataset(str(train_num)+"_n_aug_%d" % (m+1), data_ndata_aug)
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    #files.clear()
    files = []
    files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()   
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)

def generate_noise_level_data(image, image_name, out_folder):
    '''
    Generate AWGN noisy images of different levels at different channels
    Given an image, 
    B: 0-75 at 15
    G: 0-75 at 15
    R: 0-75 at 15
    totally 216 images for one input image
    '''
    os.mkdir(os.path.join(out_folder, image_name))
    for i in range(6):
       for j in range(6):
           for k in range(6):
               noise_level_list = [i * 15, j * 15, k * 15]  #the pre-defined noise level
               noisy = generate_noisy(image/255., 3, np.array(noise_level_list) / 255.)  #generate noisy image according to the given levels
               cv2.imwrite(os.path.join(out_folder, image_name, image_name + '_%d_%d_%d.png' % (i+1, j+1, k+1)), noisy[:,:,::-1]*255.)

def prepare_noise_level_data(data_path, out_path):
    # train
    files = glob.glob(os.path.join(data_path, '*'))
    for i in range(len(files)):
        file_name = files[i].split('/')[-1].split('.')[0]
        img = cv2.imread(files[i])
        img = img[:,:,::-1]
        generate_noise_level_data(img, file_name, out_path)
        
class Dataset(udata.Dataset):
    def __init__(self, c=0, train=True):
        super(Dataset, self).__init__()
        self.train = train
        self.c = c
        if self.train:
            if self.c==0:
                h5f = h5py.File('train.h5', 'r')
            elif self.c==1:
                h5f = h5py.File('train_c.h5', 'r')  
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            if self.c==0:
                h5f = h5py.File('train.h5', 'r')
            elif self.c==1:
                h5f = h5py.File('train_c.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
