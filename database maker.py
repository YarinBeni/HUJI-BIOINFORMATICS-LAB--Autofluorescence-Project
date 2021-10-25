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

BATCH_SIZE = 3  # todo: ask if needed to change (modify to 2 if sample is 2 images)
PIXEL_SIZE = 224  # todo: ask if needed to change
# mean = []
# std = []
# todo: ask if needed to get mean of dataset?
# todo: ask if needed to get std of dataset?
TRAIN_DATASET_PATH = r"C:\Users\yarin\PycharmProjects\pythonProject\tempo_dataset\database_second_iter"
# TEST_DATASET_PATH = ""  # todo: make a path for test dataset

# todo: to ask if needed customize transform if so to modified and use this section
# todo: is it sufficient to upload from cv2 as grey or also to use greyscale transform ?
# OPTIONAL train_transform = transforms.Compose([transforms.Resize((PIXEL_SIZE,PIXEL_SIZE)),transforms.ToTensor(),
# OPTIONAL transforms.Normalize(torch.Tensor(mean),std)])
# OPTIONAL test_transform = transforms.Compose([transforms.Resize((PIXEL_SIZE,PIXEL_SIZE)),transforms.ToTensor(),
# OPTIONAL transforms.Normalize(torch.Tensor(mean),std)])
train_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])


# todo: ask how to split test and train data set
# test_dataset = torchvision.datasets.ImageFolder(root = TEST_DATASET_PATH,transform=test_transform)
train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DATASET_PATH, transform=train_transform)
# dataset = WormsDataset(csv_file="", root_dir=TRAIN_DATASET_PATH, transform=train_transform)


dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# here i tried to under how my data is looking but now i can use the visualization
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

# visualization of the pictures start from here
torch.set_printoptions(linewidth=120)
# len(train_dataset)
# len(dataloader)
#lst = []


def show_images_one_by_one(dataset):
    """get the dataset object of images and print them one by one"""
    # print images one-by one from a dataset object
    # cnt = 1
    for sample in dataset:
        # print("this is iter number: ", cnt)
        # lst.append(image)
        image, label = sample
        plt.imshow(image.squeeze(), cmap="gray")
        plt.show()
    #     cnt += 1
    # print("this is number of images showed :",cnt)
    plt.close()


def show_in_grid(images_iter):
    """gets dataloader object and print in one grid all images of same sample"""
    for batch in images_iter:
        # for batch in dataloader:
        # batch = next(iter(dataloader))
        images, labels = batch
        grid = torchvision.utils.make_grid(images, nrow=3)  # padding=0,nrow and padding variables need to be adjust
        plt.figure(figsize=(15, 15))
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.show()

# UN COMMENT the action you desired:

# #this will show in one grid all 3 samples of same image
# show_in_grid(dataloader)
#
# #this will show all images one by one
# show_images_one_by_one(train_dataset)

# todo: find a way to make all picture same size than make batch bigger and show in one grid, maybe different way?
# https://stackoverflow.com/questions/61943896/pad-torch-tensors-of-different-sizes-to-be-equal
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# from here down its work in progress

def get_pad_size(dataset, pad_size=[0, 0]):
    """get the minimal contain rectangle of the a batch"""
    for sample in dataset:
        image, label = sample
        col_len = len(image[0][0])
        row_len = len(image[0])
        if col_len > pad_size[1]:
            pad_size[1] = col_len
        if row_len > pad_size[0]:
            pad_size[0] = row_len
    return pad_size


# in this data the biggest is (202,933)

# pad_size = get_pad_size(dataloader)
# max_width, max_height = pad_size

# The needed padding is the difference between the
# max width/height and the image's actual width/height.
# if this work i should have a list of the padded images with the same size

# images_iter = [F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)]) for img in dataloader]

