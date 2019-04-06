# [When AWGN-based Denoiser Meets Real Noises ]()

## Abstract
Discriminative learning based image denoisers have achieved promising performance on synthetic noise such as the additive Gaussian noise. However, their performance on images with real noise is often not satisfactory. The main reason is that real noises are mostly spatially/channel-correlated and spatial/channel-variant. In contrast, the synthetic Additive White Gaussian Noise (AWGN) adopted in most previous work is pixel-independent. In this paper, we propose a novel approach to boost the performance of a real image denoiser which is trained only with synthetic pixel-independent noise data. First, we train a deep model that consists of a noise estimator and a denoiser with mixed AWGN and Random Value Impulse Noise (RVIN). We then investigate Pixel-shuffle Down-sampling (PD) strategy to adapt the trained model to real noises. Extensive experiments demonstrate the effectiveness and generalization ability of the proposed approach. Notably, our method achieves state-of-the-art performance on real sRGB images in the DND benchmark. 

## Network Structure

![Image of Network](figs/CBDNet_v13.png)

## Realistic Noise Model
Given a clean image `x`, the realistic noise model can be represented as:

![](http://latex.codecogs.com/gif.latex?\\textbf{y}=f(M^{-1}(M(\\textbf{L}+n(\\textbf{x})))))

![](http://latex.codecogs.com/gif.latex?n(\\textbf{x})=n_s(\\textbf{x})+n_c)

Where `y` is the noisy image, `f(.)` is the CRF function which converts irradiance `L` to `x`. `M(.)` represents the function that convert sRGB image to Bayer image and `M^(-1)(.)` represents the demosaicing function.

If considering denosing on compressed images, 

![](http://latex.codecogs.com/gif.latex?\\textbf{y}=JPEG(f(M^{-1}(M(\\textbf{L}+n(\\textbf{x}))))))

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

------------------------------------------------------------------------------------------------------------------------------

**Q&A:** Why CBDNet can not process some high-noisy photos captured by my own?

A: The main reason is the JPEG compression. For uncompression images even with really high noise under low light condition, CBDNet can remove noise very effectively. Even though we consider JPEG compression on CBDNet, our CBDNet(JPEG) model can only handle jpeg images with normal noise level, e.g., Nam or JPEG compression quality is high.   

I capture some high-noisy images using DLSR camera. Images are stored in both *uncompressed* and *JPEG* format. The denoising results are shown below. 
![](figs/results.png)
