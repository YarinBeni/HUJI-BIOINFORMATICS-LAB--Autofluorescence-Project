import torch
import torchvision
import torchvision.transforms as transforms
# import os
import numpy as np
import matplotlib.pyplot as plt
# import cv2 as cv
import torch.utils.data as data
import torchvision.transforms.functional as F
from Worms_Dataset import WormsDataset

############################################################################################################
#  this file was split into Worms_Dataset & preprocessing_Crop_and_rotate files
#  and is no longer violable maybe will find use in future
############################################################################################################












# BATCH_SIZE = 3  # todo: ask if needed to change (modify to 2 if sample is 2 images)
# PIXEL_SIZE = 224  # todo: ask if needed to change
# # mean = []
# # std = []
# # todo: ask if needed to get mean of dataset?
# # todo: ask if needed to get std of dataset?
# TRAIN_DATASET_PATH = r"C:\Users\yarin\PycharmProjects\pythonProject\tempo_dataset\database_second_iter"
# # TEST_DATASET_PATH = ""  # todo: make a path for test dataset
#
# # todo: to ask if needed customize transform if so to modified and use this section
# # todo: is it sufficient to upload from cv2 as grey or also to use greyscale transform ?
# # OPTIONAL train_transform = transforms.Compose([transforms.Resize((PIXEL_SIZE,PIXEL_SIZE)),transforms.ToTensor(),
# # OPTIONAL transforms.Normalize(torch.Tensor(mean),std)])
# # OPTIONAL test_transform = transforms.Compose([transforms.Resize((PIXEL_SIZE,PIXEL_SIZE)),transforms.ToTensor(),
# # OPTIONAL transforms.Normalize(torch.Tensor(mean),std)])
# train_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
#
#
# # todo: ask how to split test and train data set
# # test_dataset = torchvision.datasets.ImageFolder(root = TEST_DATASET_PATH,transform=test_transform)
# train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DATASET_PATH, transform=train_transform)
# # dataset = WormsDataset(csv_file="", root_dir=TRAIN_DATASET_PATH, transform=train_transform)
#
#
# dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# here i tried to understand how my data is looking but now i can use the visualization
# print("this is len of dataloader:", len(dataloader))
# cnt = 1
# lst = []
# while cnt < 4:
#     print("this is batch number:", cnt)
#     images, labels = next(iter(dataloader))
#     print("this is images len : ", len(images))
#     print("this is images size: ", images.size())
#
#     print("this is images[0] size: ", images[0].size())
#     # print("this is images[0]: ")
#     # print(images[0])
#     print()
#     cntt = 1
#     print()
#     print()
#     print()
#     lst.append(id(images[0]))
#     images, labels = next(iter(dataloader))
#     for data in images[0]:
#         print("this is data size in images[0]", data.size())
#         print("this is data type: in images[0]", type(data))
#         print("this is data from images, picture number: ", cntt)
#         cntt += 1
#         print()
#         # print(data)
#     print("#############################")
#     print()
#     cnt += 1
# x = lst[0]
# for a in lst[1:]:
#     if a == x:
#         print("same")
# print("not same")
#

# from here down its work in progress


