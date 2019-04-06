# [When AWGN-based Denoiser Meets Real Noises ]()

## Abstract
Discriminative learning based image denoisers have achieved promising performance on synthetic noise such as the additive Gaussian noise. However, their performance on images with real noise is often not satisfactory. The main reason is that real noises are mostly spatially/channel-correlated and spatial/channel-variant. In contrast, the synthetic Additive White Gaussian Noise (AWGN) adopted in most previous work is pixel-independent. In this paper, we propose a novel approach to boost the performance of a real image denoiser which is trained only with synthetic pixel-independent noise data. First, we train a deep model that consists of a noise estimator and a denoiser with mixed AWGN and Random Value Impulse Noise (RVIN). We then investigate Pixel-shuffle Down-sampling (PD) strategy to adapt the trained model to real noises. Extensive experiments demonstrate the effectiveness and generalization ability of the proposed approach. Notably, our method achieves state-of-the-art performance on real sRGB images in the DND benchmark. 

## Network Structure

![Image of Network](figs/CBDNet_v13.png)

## Pixel-Shuffle Down-sampling

## Testing
* "Test_Patches.m" is the testing code for small images or image patches. If the tesing image is too large (e.g., 5760*3840), we recommend to use "Test_fullImage.m"
*  "Test_fullImage.m" is the testing code for large images. 
*  "Test_Realistic_Noise_Model.m" is the testing code for the realistic noise mode in our paper. And it's very convinent to utilize [AddNoiseMosai.m](https://github.com/GuoShi28/CBDNet/blob/master/utils/AddNoiseMosai.m) to train your own denoising model for real photographs.

## CBDNet Models
* "CBDNet.mat" is the testing model for DND dataset and NC12 dataset for not considering the JPEG compression.
*  "CBDNet_JPEG.mat" is the testing model for Nam dataset and other noisy images with JPEG format.

## Implement Perceptual Loss Using MatConvnet
The perceptual loss is the MSE loss between the [Perceptual Layer](https://github.com/GuoShi28/CBDNet/tree/master/utils/Perceptual_Layer) outputs of results and labels.
The pretrained vgg model, [fast-rcnn-vgg16-pascal07-dagnn](http://www.vlfeat.org/matconvnet/pretrained/) is needed. 

## Real Images Denoising Results
### DND dataset
Following the guided of [DND Online submission system](https://noise.visinf.tu-darmstadt.de/).

![Image of DND](figs/DND_results.png)

### Nam dataset

![Image of Nam](figs/Nam_results.png)

## Requirements and Dependencies
* Matlab 2015b
* Cuda-8.0 & cuDNN v-5.1
* [MatConvNet](http://www.vlfeat.org/matconvnet/).

## Citation
arxiv: [https://arxiv.org/abs/1807.04686](https://arxiv.org/abs/1807.04686)
```
@article{Guo2019Cbdnet,
  title={Toward convolutional blind denoising of real photographs},
  author={Guo, Shi and Yan, Zifei and Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={2019 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
# DnCNN-PyTorch

This is a PyTorch implementation of the TIP2017 paper [*Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*](http://ieeexplore.ieee.org/document/7839189/). The author's [MATLAB implementation is here](https://github.com/cszn/DnCNN).

****
This code was written with PyTorch<0.4, but most people must be using PyTorch>=0.4 today. Migrating the code is easy. Please refer to [PyTorch 0.4.0 Migration Guide](https://pytorch.org/blog/pytorch-0_4_0-migration-guide/).

****

## How to run

### 1. Dependences
* [PyTorch](http://pytorch.org/)(<0.4)
* [torchvision](https://github.com/pytorch/vision)
* OpenCV for Python
* [HDF5 for Python](http://www.h5py.org/)
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) (TensorBoard for PyTorch)

### 2. Train DnCNN-S (DnCNN with known noise level)
```
python train.py \
  --preprocess True \
  --num_of_layers 17 \
  --mode S \
  --noiseL 25 \
  --val_noiseL 25
```
**NOTE**
* If you've already built the training and validation dataset (i.e. train.h5 & val.h5 files), set *preprocess* to be False.
* According to the paper, DnCNN-S has 17 layers.
* *noiseL* is used for training and *val_noiseL* is used for validation. They should be set to the same value for unbiased validation. You can set whatever noise level you need.

### 3. Train DnCNN-B (DnCNN with blind noise level)
```
python train.py \
  --preprocess True \
  --num_of_layers 20 \
  --mode B \
  --val_noiseL 25
```
**NOTE**
* If you've already built the training and validation dataset (i.e. train.h5 & val.h5 files), set *preprocess* to be False.
* According to the paper, DnCNN-B has 20 layers.
* *noiseL* is ingnored when training DnCNN-B. You can set *val_noiseL* to whatever you need.

### 4. Test
```
python test.py \
  --num_of_layers 17 \
  --logdir logs/DnCNN-S-15 \
  --test_data Set12 \
  --test_noiseL 15
```
**NOTE**
* Set *num_of_layers* to be 17 when testing DnCNN-S models. Set *num_of_layers* to be 20 when testing DnCNN-B model.
* *test_data* can be *Set12* or *Set68*.
* *test_noiseL* is used for testing. This should be set according to which model your want to test (i.e. *logdir*).

## Test Results

### BSD68 Average RSNR

| Noise Level | DnCNN-S | DnCNN-B | DnCNN-S-PyTorch | DnCNN-B-PyTorch |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|
|     15      |  31.73  |  31.61  |      31.71      |      31.60      |
|     25      |  29.23  |  29.16  |      29.21      |      29.15      |
|     50      |  26.23  |  26.23  |      26.22      |      26.20      |

### Set12 Average PSNR

| Noise Level | DnCNN-S | DnCNN-B | DnCNN-S-PyTorch | DnCNN-B-PyTorch |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|
|     15      | 32.859  | 32.680  |     32.837      |     32.725      |
|     25      | 30.436  | 30.362  |     30.404      |     30.344      |
|     50      | 27.178  | 27.206  |     27.165      |     27.138      |

## Tricks useful for boosting performance
* Parameter initialization:  
Use *kaiming_normal* initialization for *Conv*; Pay attention to the initialization of *BatchNorm*
```
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)
```
* The definition of loss function  
Set *size_average* to be False when defining the loss function. When *size_average=True*, the **pixel-wise average** will be computed, but what we need is **sample-wise average**.
```
criterion = nn.MSELoss(size_average=False)
```
The computation of loss will be like:
```
loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
```
where we divide the sum over one batch of samples by *2N*, with *N* being # samples.
