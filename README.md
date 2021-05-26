# multimedia_term_project

#### This project is part of the 2021 multimedia class ConvNet challenge.
http://cvipcc.com/leaderboard/colorization

## Contents
1. Data description
2. Model descrition
3. Terms of use

## Data description
The dataset was provided by the challenge. The dataset is consisted of 5000 images. Training and validation data is split with a 9 to 1 ratio. Testing data is consisted of 2 seperate files consisting of lightness images of channel 1 and mask images of channel 2.

## Model description
Our propsed model uses a mix of Unet, Resblocks and attention networks.

#### Unet
![image](https://user-images.githubusercontent.com/51948435/119621354-facf3500-be40-11eb-8177-19db9b5a1087.png)
U-Net: Convolutional Networks for Biomedical Image Segmentation https://arxiv.org/abs/1505.04597<br>
Unet was first proposed by Olaf.R et.al for biomedical image segmentation. Due to its versitility it is also used in various fields including image colorization.<br>
![image](https://github.com/sungyoonahn/multimedia_term_project/blob/main/mynet2.png)
Our Unet is consists of 5 layes which includes 5 encoding layers and 4 decoding layers.<br>
Each encoding layer consists of a 2 convolution layers with a residual block followed by a maxpooling layer.<br>
Each decoding layer consists of an attention layer followed by a transopse convolution layer of upscaling and skip-connection from the previous layer followed by a double convilution layer with a residual block.<br>


