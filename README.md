# Autofluorescence Solution using Computer Vision and Deep Learning
## Introduction
This project aims to address the problem of autofluorescence in images using computer vision and deep learning techniques.  
Autofluorescence is a phenomenon that occurs when certain materials emit light in response to excitation by external light sources.  
This can cause unwanted artifacts in images and can make it difficult to accurately analyze or interpret the content of the image.

<img src="https://github.com/YarinBeni/HUJI-BIOINFORMATICS-LAB--Autofluorescence-Project/blob/main/Models%20Input%2C%20Label%2C%20Prediction.png?raw=true" width="600" height="200">

*Figure: UNet Model Visualization. From left to right - the first image shows the partial input to the model, which consists of a tuple comprising a standard image and its corresponding X-ray image, here just the standard image. The middle image displays the target, which is the same image devoid of autofluorescence noise. The last image on the right is the model's output, a prediction made by the UNet model to closely replicate the signal in the target image, minimizing the influence of noise.*

## Methodology
To tackle this problem, I used a local database of images , built a custom dataset for the PyTorch API to use for training and implemented the U-Net deep learning model with PyTorch and OpenCV.   

The U-Net model is a popular choice for image segmentation tasks, as it is able to effectively capture both the local and global context of an image.  
It is composed of an encoder and a decoder, with skip connections between the two to allow for the transfer of low-level features from the encoder to the decoder.
From Paper "U-Net: Convolutional Networks for Biomedical Image Segmentation", Olaf Ronneberger, Philipp Fischer, Thomas Brox link: https://arxiv.org/abs/1505.04597.  

## Current Status
Unfortunately, this project was stopped in the middle of the training stage due to unforeseen circumstances. 

## Acknowledgements
I would like to extend my heartfelt thanks to Dr. Remer and Dr. Ben David from Agilent Technologies Research Labs and Dr. Ben David from Hebrew University for their invaluable mentorship and guidance on this project.  
Their expertise and support were instrumental in the development of this project.




