# Hint-based-image-colorization-using-Attention-Unet

#### This project is part of the 2021 multimedia class ConvNet challenge.
<!-- http://cvipcc.com/leaderboard/colorization -->

## Contents
1. Data description
2. Model description
3. How to run the code

## 1. Data description
The dataset was provided by the challenge. The dataset is consisted of 5000 images. Training and validation data is split with a 9 to 1 ratio. Testing data is consisted of 2 seperate files consisting of lightness images of channel 1 and mask images of channel 2.

## 2. Model description
Our propsed model uses a mix of Unet, Resblocks and attention networks.

### Related work

#### Resblock
![image](https://github.com/sungyoonahn/multimedia_term_project/blob/main/resblock.png)<br>
Deep Residual Learning for Image Recognition https://arxiv.org/abs/1512.03385<br>
Resblocks were proposed to enhance a models performance by concatenating featuremaps from the previous convolution to the current one. This way some important features are preserved.

#### Unet
![image](https://user-images.githubusercontent.com/51948435/119621354-facf3500-be40-11eb-8177-19db9b5a1087.png)<br>
U-Net: Convolutional Networks for Biomedical Image Segmentation https://arxiv.org/abs/1505.04597<br>
Unet was first proposed by Olaf.R et.al for biomedical image segmentation. Due to its versitility it is also used in various fields including image colorization.

#### Attention Unet
[image](https://github.com/sungyoonahn/multimedia_term_project/blob/main/attention_layer.png)<br>
![image](https://user-images.githubusercontent.com/51948435/119641508-b5b4fe00-be54-11eb-9757-6f54edaca72c.png)
Attention U-Net: Learning Where to Look for the Pancreas https://arxiv.org/abs/1804.03999<br>
The attention gate was proposed to enhace the performance for Unet used in segmenting the pancreas. As shown in the image above, what an attention gate does is it takes the encoding layer featuremaps through skip connection and concatenates it with the featuremap from the previous decoding layer. It then is put through a sigmoid function. 

### Our proposed network

![image](https://github.com/sungyoonahn/multimedia_term_project/blob/main/mynet2.png)<br>
<!-- Out model is a mix of the classic attention-unet with additional resblocks and a slightly different input by combining featuremaps from the attention gate, the d -->
Our Unet is consists of 5 layes which includes 5 encoding layers and 4 decoding layers.<br>
Each encoding layer consists of a 2 convolution layers with a residual block followed by a maxpooling layer.<br>
Each decoding layer consists of an attention layer followed by a tranpose convolution layer and skip-connection from the previous layer followed by a double convolution layer with a residual block.<br>

#### results
![image](https://github.com/sungyoonahn/multimedia_term_project/blob/main/colorization_output.png)<br>

## 3. How to use the code
- dataloader.py contains codes for creating hint images and loading data.
- main.py contains codes for training the model.
- mynet2.py containes codes for the attention-unet
- test.py containes codes for testing test images provided by the contest.
- train.py contains codes for train, validation and test functions.
- utils.py contains codes for saving validation outputs, calculating loss, calculating psnr and ssim values.

run main.py<br>
Good Luck!
